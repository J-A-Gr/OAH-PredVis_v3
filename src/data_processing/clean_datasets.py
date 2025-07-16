import pandas as pd
import os

# Directories
SRC_DIR = "data/raw"
DST_DIR = "data/cleaned"
os.makedirs(DST_DIR, exist_ok=True)

# 1. Clean Customers
customers = pd.read_csv(
    os.path.join(SRC_DIR, "oahdotnet_live_Customers.csv"),
    parse_dates=["UpdatedOn"],
    low_memory=False
)
customers_clean = customers[["Id", "PicqerId", "LifetimeOrders", "LifetimeValue"]]
customers_clean.to_csv(os.path.join(DST_DIR, "customers_clean.csv"), index=False)

# 2. Clean Orders
orders = pd.read_csv(
    os.path.join(SRC_DIR, "oahdotnet_live_Orders.csv"),
    parse_dates=["CreatedOn"],
    low_memory=False
)
orders_clean = orders[[
    # Essential columns
    "Id",
    "OrdersAfkomstId",
    "IdCustomer",
    "CreatedOn",
    "TotalPrice",
    "Status",
    # "DiscountCode", # NaN >40%
    "ShippingAmount",
    # "ShippingTaxAmount",   # NaN >40%
    # "ShippingIncludedTaxes",  # NaN >40%
    # Nice-to-have enrichments
    "DeliveryName",
    "DeliveryAddress",
    "DeliveryZipcode",
    "DeliveryCity",
    "DeliveryCountry",
    # "InvoiceCountry",
    # "FullInvoiceAddress",  # NaN >40%
    "TotalProducts",
    "TotalPicked",
    "TotalPicklists",
    "PicklistId",
    "ShippingIsUpdated",
    # "ShippingProvider",  # NaN >40%
    # "ShippingLabelUpdatedOn",   # NaN >40%
    "Reference",
    # "ExternalApiSource", #  # NaN >40%
    "MaP",
    "IdTemplate",
    "JaarOrder",
    "JaarOrderNr"
]]

orders_clean.to_csv(os.path.join(DST_DIR, "orders_clean.csv"), index=False)

# 3. Clean OrderItems
items = pd.read_csv(
    os.path.join(SRC_DIR, "oahdotnet_live_OrdersItem.csv"),
    parse_dates=["CreatedOn"],
    low_memory=False
)
items_clean = items[[
    # Crucial
    "OrdersId",
    "ProductId",
    "Amount",
    "Price",
    # Nice-to-have
    "VatgroupId",
    # "DefaultLocationsId",  # NaN >40%
    # "LocationName",   # NaN >40%
    "HasParts",
    # "AmountCancelled",
    "Weight",
    "CreatedOn"
]]
items_clean.to_csv(os.path.join(DST_DIR, "order_items_clean.csv"), index=False)

# 4. Clean Products
products = pd.read_csv(
    os.path.join(SRC_DIR, "oahdotnet_live_Products.csv"),
    low_memory=False
)

products_clean = products[[
    # 🔑 Crucial
    "Id",
    "Price",
    "TotalStock",
    "ProductCode",  # change name to SKU
    # 👍 Nice-to-have
    "Name",
    "Category",
    # "PcsSold365Days",  # NaN >40%
    # "Predict30dPcs",  # NaN >40% 
    # "DaysToEmptyStock",  # NaN >40%
    "ToBuy",
    # "ToBuyValue",  # NaN >40%
    "Weight",
    "SupplierName",
    "DeliveryTime"
]]

products_clean.to_csv(os.path.join(DST_DIR, "products_clean.csv"), index=False)

# 5. Clean Stock
stock = pd.read_csv(
    os.path.join(SRC_DIR, "oahdotnet_live_ProductsStock.csv"),
    low_memory=False
)
# Adjust column names if necessary ('ProductsId' vs 'ProductId')

stock_clean = stock[[
    # 🔑 Crucial
    "ProductsId",
    "Freestock",
    # (optional) "Stock",
    # 👍 Nice-to-have
    # "Reserved",
    # "ReservedBackorders",
    # "ReservedPicklists",
    # "ReservedAllocations",
    "WarehousesId"
]]

# Then save:
stock_clean.to_csv("data/cleaned/stock_clean.csv", index=False)

stock_clean.to_csv(os.path.join(DST_DIR, "stock_clean.csv"), index=False)

# # 6. Clean StockHistory
# stock_hist = pd.read_csv(
#     os.path.join(SRC_DIR, "oahdotnet_live_StockHistory.csv"),
#     parse_dates=["UpdatedOn"],
#     low_memory=False
# )
# # Keep only product reference, quantity change, and timestamp
# stock_hist_clean = stock_hist[["ProductsId", "QuantityChange", "UpdatedOn"]]
# stock_hist_clean.to_csv(os.path.join(DST_DIR, "stock_history_clean.csv"), index=False)

print("All datasets cleaned and saved to 'data/cleaned/'")
