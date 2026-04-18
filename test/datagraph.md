# Data Relationship Diagram

This diagram shows the main links between the CSV files in the `datathon` dataset.

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px', 'primaryColor': '#fef6dd', 'secondaryColor': '#d1e8ff', 'tertiaryColor': '#f8f8ff', 'edgeLabelBackground': '#ffffff', 'lineColor': '#333', 'textColor': '#111', 'nodeBorderRadius': '8px'}}}%%
flowchart LR
  subgraph Transactions
    train_tx["train_transaction.csv"]
    test_tx["test_transaction.csv"]
  end

  subgraph Identity
    train_id["train_identity.csv"]
    test_id["test_identity.csv"]
  end

  train_tx -->|TransactionID| train_id
  test_tx -->|TransactionID| test_id

  train_tx --> TransactionAmt["TransactionAmt"]
  train_tx --> ProductCD["ProductCD"]
  train_tx --> card1["card1"]
  train_tx --> card2["card2"]
  train_tx --> card3["card3"]
  train_tx --> card4["card4"]
  train_tx --> card5["card5"]
  train_tx --> card6["card6"]
  train_tx --> addr1["addr1"]
  train_tx --> addr2["addr2"]
  train_tx --> P_emaildomain["P_emaildomain"]
  train_tx --> R_emaildomain["R_emaildomain"]
  train_tx --> dist1["dist1"]
  train_tx --> dist2["dist2"]
  train_tx --> isFraud["isFraud"]

  train_id --> id_01["id_01..id_11"]
  train_id --> id_12["id_12..id_38"]
  train_id --> DeviceType["DeviceType"]
  train_id --> DeviceInfo["DeviceInfo"]

  test_tx --> TransactionAmt
  test_tx --> ProductCD
  test_tx --> card1
  test_tx --> card2
  test_tx --> card3
  test_tx --> card4
  test_tx --> card5
  test_tx --> card6
  test_tx --> addr1
  test_tx --> addr2
  test_tx --> P_emaildomain
  test_tx --> R_emaildomain
  test_tx --> dist1
  test_tx --> dist2

  test_id --> id_01
  test_id --> id_12
  test_id --> DeviceType
  test_id --> DeviceInfo
```

## How to view
- Open this file in VS Code
- Use Markdown preview (`Ctrl+Shift+V`)
- If needed, install `Markdown Preview Mermaid Support` extension