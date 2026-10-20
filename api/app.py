from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "products.csv"
CSV_FIELDS = ["id", "name", "quantity", "unit"]

app = FastAPI(
    title="Basic Inventory API",
    description="CSV-backed inventory service used by the AI inventory agent.",
    version="1.0.0",
)


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    quantity: float = Field(ge=0)
    unit: str = Field(min_length=1, max_length=30)

    @field_validator("name", "unit")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class StockUpdate(BaseModel):
    delta: float

    @field_validator("delta")
    @classmethod
    def delta_must_not_be_zero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("delta must be positive for incoming stock or negative for outgoing stock")
        return value


class Product(BaseModel):
    id: int
    name: str
    quantity: float
    unit: str


def ensure_csv_exists() -> None:
    """Create the CSV with a header when it does not exist yet."""
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()


def read_products() -> list[dict]:
    ensure_csv_exists()
    products: list[dict] = []

    with CSV_PATH.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                products.append(
                    {
                        "id": int(row["id"]),
                        "name": row["name"],
                        "quantity": float(row["quantity"]),
                        "unit": row["unit"],
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Invalid data found in products.csv: {exc}",
                ) from exc

    return products


def write_products(products: list[dict]) -> None:
    """Persist the full current inventory back to products.csv."""
    with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(products)


def next_product_id(products: list[dict]) -> int:
    return max((product["id"] for product in products), default=0) + 1


@app.get("/", tags=["health"])
def root() -> dict:
    return {"message": "Inventory API is running."}


@app.get("/inventory", response_model=list[Product], tags=["inventory"])
def list_inventory() -> list[dict]:
    """Return the full product list."""
    return read_products()


@app.post(
    "/inventory",
    response_model=Product,
    status_code=status.HTTP_201_CREATED,
    tags=["inventory"],
)
def add_product(product: ProductCreate) -> dict:
    """Add a new product to the inventory."""
    products = read_products()

    duplicate = next(
        (item for item in products if item["name"].casefold() == product.name.casefold()),
        None,
    )
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Product '{product.name}' already exists with id {duplicate['id']}. "
                "Use PATCH /inventory/{product_id} to change its stock."
            ),
        )

    new_product = {
        "id": next_product_id(products),
        "name": product.name,
        "quantity": product.quantity,
        "unit": product.unit,
    }
    products.append(new_product)
    write_products(products)
    return new_product


@app.patch("/inventory/{product_id}", response_model=Product, tags=["inventory"])
def update_stock(product_id: int, update: StockUpdate) -> dict:
    """Apply a stock delta: positive for incoming, negative for outgoing."""
    products = read_products()

    for product in products:
        if product["id"] == product_id:
            new_quantity = product["quantity"] + update.delta
            if new_quantity < 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Insufficient stock for '{product['name']}'. "
                        f"Current quantity is {product['quantity']} {product['unit']}; "
                        f"requested delta is {update.delta}."
                    ),
                )

            product["quantity"] = new_quantity
            write_products(products)
            return product

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Product with id {product_id} was not found.",
    )


@app.get("/inventory/alerts", response_model=list[Product], tags=["inventory"])
def low_stock_alerts(
    threshold: Annotated[
        float,
        Query(ge=0, description="Products below this quantity are considered low stock."),
    ] = 10,
) -> list[dict]:
    """Return products whose quantity is below the supplied threshold."""
    products = read_products()
    return [product for product in products if product["quantity"] < threshold]
