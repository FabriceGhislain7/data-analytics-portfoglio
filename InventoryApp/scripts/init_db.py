from inventory_app.db import engine
from inventory_app.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)
    print("tables created successfully")

if __name__ == "__main__":
    init_db()