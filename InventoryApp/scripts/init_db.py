from inventory_app.db import engine
from inventory_app.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully")
