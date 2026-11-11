import pandas as pd
import numpy as np

fact = pd.read_csv("fact_sales.csv")
dim_time = pd.read_csv("dim_time.csv")
dim_product = pd.read_csv("dim_product.csv")
dim_location = pd.read_csv("dim_location.csv")
for name, t in [("fact_sales", fact), ("dim_time", dim_time), ("dim_product", dim_product), ("dim_location", dim_location)]:
    print(name, t.shape)
n_lines = len(fact)
n_orders = fact["order_id"].nunique()
print("мөр:", n_lines, "захиалга:", n_orders)
print(fact["time_key"].isin(dim_time["time_key"]).all())
print(fact["product_key"].isin(dim_product["product_key"]).all())
print(fact["location_key"].isin(dim_location["location_key"]).all())
dw = (fact
      .merge(dim_time, on="time_key", validate="many_to_one")
      .merge(dim_product, on="product_key", validate="many_to_one")
      .merge(dim_location, on="location_key", validate="many_to_one"))
print(len(dw) == len(fact), dw["amount"].sum() == fact["amount"].sum())
print(dw.columns.tolist())
dim_location_dup = pd.concat([dim_location, dim_location.iloc[[0]]], ignore_index=True)
fan = fact.merge(dim_time, on="time_key").merge(dim_product, on="product_key").merge(dim_location_dup, on="location_key")
print("fan-out мөр:", len(fan), "нийлбэр:", fan["amount"].sum(), "анхны:", fact["amount"].sum())
levels = {"time": 4, "product": 2, "location": 3}
total = 1
for L in levels.values():
    total *= (L + 1)
print("нийт cuboid:", total)
levels2 = dict(levels, product=3)
print("brand нэмвэл:", np.prod([L + 1 for L in levels2.values()]))
total_amt = dw["amount"].sum()
cuboids = {
    "apex": [],
    "1-D": ["year"],
    "2-D": ["year", "category"],
    "3-D": ["year", "category", "region"],
    "base": ["date", "product_name", "store"],
}
rows = []
for name, keys in cuboids.items():
    if keys:
        g = dw.groupby(keys)["amount"].sum()
        cells, s = len(g), g.sum()
    else:
        cells, s = 1, total_amt
    rows.append((name, cells, s))
print(pd.DataFrame(rows, columns=["cuboid", "cells", "sum_amount"]))
time_levels = {
    "өдөр": ["date"],
    "сар": ["year", "month"],
    "улирал": ["year", "quarter"],
    "жил": ["year"],
}
out = []
for name, keys in time_levels.items():
    g = dw.groupby(keys)["amount"].sum()
    out.append((name, len(g), g.sum()))
print(pd.DataFrame(out, columns=["түвшин", "мөр", "нийлбэр"]))
wrong = dw.groupby("month")["amount"].sum()
right = dw.groupby(["year", "month"])["amount"].sum()
print(len(wrong), len(right))
print("1-р сар (буруу):", wrong.loc[1])
print("1-р сар 2024 + 2025:", right.loc[(2024, 1)] + right.loc[(2025, 1)])
print(wrong.sum() == right.sum())
loc_levels = {"store": ["store"], "city": ["city"], "region": ["region"]}
for name, keys in loc_levels.items():
    g = dw.groupby(keys)["amount"].sum()
    print(name, len(g), g.sum())
by_city = dw.groupby("city")["amount"].sum().sort_values(ascending=False)
print(by_city)
cube3 = dw.groupby(["year", "category", "region"])["amount"].sum()
via_cube = cube3.groupby(level=["year", "category"]).sum()
direct = dw.groupby(["year", "category"])["amount"].sum()
print(np.allclose(via_cube, direct))
print(direct.unstack())
d25 = dw[dw["year"] == 2025]
reg = d25.groupby("region")["amount"].sum().sort_values(ascending=False)
top_region = reg.index[0]
print(top_region, round(reg.iloc[0] / reg.sum() * 100, 1), "%")
in_region = d25[d25["region"] == top_region]
cty = in_region.groupby("city")["amount"].sum().sort_values(ascending=False)
cty_tbl = cty.to_frame("amount")
cty_tbl["share_pct"] = (cty_tbl["amount"] / cty_tbl["amount"].sum() * 100).round(1)
print(cty_tbl)
top_city = cty.index[0]
sto = in_region[in_region["city"] == top_city].groupby("store")["amount"].sum().sort_values(ascending=False)
print(sto)
qsum = d25.groupby("quarter")["amount"].sum()
top_q = qsum.idxmax()
print(qsum, "-> хамгийн их:", top_q)
msum = d25[d25["quarter"] == top_q].groupby("month")["amount"].sum()
print(msum, "-> хамгийн их сар:", msum.idxmax())
print("slice:", d25["amount"].sum())
dc = dw[
    (dw["year"] == 2025)
    & (dw["region"].isin(["Төв", "Баруун"]))
    & (dw["category"].isin(["Утас", "Ноутбук"]))
    & (dw["quarter"].isin([3, 4]))
]
print("dice:", len(dc), dc["amount"].sum())
pv = pd.pivot_table(d25, index="region", columns="quarter", values="amount",
                    aggfunc="sum", margins=True, margins_name="Нийт")
print(pv)
print(pv.loc["Нийт", "Нийт"] == d25["amount"].sum())
print(pv.T)

sy = pd.pivot_table(dw, index="store", columns="year", values="amount", aggfunc="sum")
print(sy)
print(sy.fillna(0))
print("эхний борлуулалт:", dw.loc[dw["store"] == "Улаангом салбар", "date"].min())

y24 = dw[dw["year"] == 2024]
true_avg = y24["amount"].mean()
by_q = y24.groupby("quarter")["amount"].agg(["mean", "sum", "count"])
wrong_avg = by_q["mean"].mean()
print(round(true_avg, 2), round(wrong_avg, 2))
alg_avg = by_q["sum"].sum() / by_q["count"].sum()
print(round(alg_avg, 2), np.isclose(alg_avg, true_avg))

true_med = y24["amount"].median()
q_med = y24.groupby("quarter")["amount"].median()
wrong_med = q_med.median()
print(true_med, wrong_med, true_med == wrong_med)

g = dw.groupby("category")
simple = g["unit_price"].mean()
weighted = g["amount"].sum() / g["quantity"].sum()
print(pd.DataFrame({"энгийн": simple.round(2), "жигнэсэн": weighted.round(2)}))

dw["profit"] = dw["amount"] - dw["cost"]
cat = dw.groupby("category").agg(amount=("amount", "sum"), profit=("profit", "sum"))
cat["margin_pct"] = (cat["profit"] / cat["amount"] * 100).round(1)
print(cat.sort_values("amount", ascending=False))

yq = dw.pivot_table(index="quarter", columns="year", values="amount", aggfunc="sum")
yq["growth_pct"] = ((yq[2025] / yq[2024] - 1) * 100).round(1)
print(yq)
print(yq["growth_pct"].idxmax(), yq["growth_pct"].idxmin())

store25 = d25.groupby("store")["amount"].sum()
print(store25.nlargest(3))
share = (d25.groupby("region")["amount"].sum() / d25["amount"].sum() * 100).round(1)
print(share.sort_values(ascending=False))
print(round(share.sum(), 1))

print(dw.groupby("year")["amount"].sum(), dw["amount"].sum())
cy = dw.pivot_table(index="category", columns="year", values="amount", aggfunc="sum")
cy["change"] = cy[2025] - cy[2024]
print(cy.sort_values("change", ascending=False))
top_reg = d25.groupby("region")["amount"].sum().idxmax()
print(d25[d25["region"] == top_reg].groupby("store")["amount"].sum().sort_values().head(1))
day = dw.groupby("date")["amount"].sum()
print(day.idxmax(), day.max())
per_order = dw.groupby("order_id")["quantity"].sum()
print(round(per_order.mean(), 2))