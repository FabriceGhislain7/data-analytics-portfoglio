from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
import datetime

Base = declarative_base()

# models for client
class Client(Base):
    """Represents a customer who can place orders."""
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    orders = relationship("Order", back_populates="client")

class Product(Base):
    """Represents a product sold by the store."""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    quantity_in_stock = Column(Integer, nullable=False)

class Order(Base):
    """Represents a customer's order."""
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    create_at = Column(DateTime, default=datetime.datetime.utcnow)
    client = relationship("Client", back_populates="orders")
