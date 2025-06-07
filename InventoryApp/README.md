
# InventoryApp

InventoryApp is a simple stock and sales management application built with Python, SQLAlchemy, and eventually Flask.  
It is designed for learning and prototyping data-driven applications with a commercial context.

## Index

- [Project Setup](#project-setup)
- [Project Structure](#project-structure)
- [Features](#features)
- [Database Models](#database-models)
- [CLI Usage](#cli-usage)
- [Future Web Interface (Flask)](#future-web-interface-flask)
- [Development Notes & Issues](#development-notes--issues)

## Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/data-analytics-portfolio.git
   cd data-analytics-portfolio/InventoryApp
   ```

2. Create and activate the virtual environment:

   ```bash
   py -m venv venv
   .\venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   Install SQLAlchemy using this command:

   ```bash
   pip install SQLAlchemy
   ```

   If there's an issue, install it directly using the virtual environment:

   ```bash
   .\venv\Scripts\python.exe -m pip install SQLAlchemy
   ```
   Check the list of program in your virtual environment
   ```bash
   pip3 list
   ```

4. Run initial database setup:

   ```bash
   python main.py
   ```

## Project Structure

```
InventoryApp/
│
├── inventory_app/                # Main application package
│   ├── __init__.py               # Package initializer
│   ├── db.py                     # Database engine and session config
│   ├── models.py                 # ORM models (Client, Product, Sale, etc.)
│   ├── crud.py                   # Business logic and data access
│   ├── cli.py                    # Command-line interface (argparse)
│   └── web/                      # (Optional) Flask app components
│       ├── __init__.py
│       ├── routes.py
│       ├── templates/
│       └── static/
│
├── scripts/
│   └── init_db.py                # Script to initialize database
│
├── main.py                       # Entry point for CLI or web server
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files and folders to exclude from Git
└── README.md                     # Project documentation
```

## Features

* Manage products, clients, and sales
* Track product quantities and availability
* Record client orders with multiple products
* Track stock usage and customer purchase history

## Database Models

* Client: name, email
* Product: name, price, quantity in stock
* Sale: linked to one client
* SaleLine: many-to-many relationship between sale and products with quantity and unit price

## CLI Usage (planned)

```bash
# Add a new product
python app.py add_product "USB Cable" 7.99 100

# Add a client
python app.py add_client "Alice Martin" "alice@example.com"

# Create a sale and add products to it
python app.py create_sale 1
python app.py add_product_to_sale 1 3 2  # sale_id, product_id, quantity
```

## Future Web Interface (Flask)

Later in the project, a simple web interface will be added using Flask:

* View all products and stock levels
* Manage clients and view purchase history
* Add sales via web form
* Filter sales by date or client

## Development Notes & Issues

This section logs personal setup notes, problems, and solutions during development:

* PowerShell pip error:
  Had to use `.\venv\Scripts\python.exe -m pip install SQLAlchemy` instead of plain `pip` due to a profile conflict.

* Python not recognized:
  Fixed by installing Python 3.10 and removing old `C:\Python39` paths in `PowerShell_profile.ps1`.

Add any new issues or configurations below:

```
