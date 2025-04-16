from typing import List, Dict
PRODUCT_CATALOG = {
    "Groceries": {
        "Rice_Grains": [
            {"id": "RG1", "name": "Basmati Rice Premium", "base_price": 125.0, "price": 125.0, "inventory": 500, "category": "Rice_Grains", "subcategory": "Premium"},
            {"id": "RG2", "name": "Brown Rice Organic", "base_price": 95.0, "price": 95.0, "inventory": 300, "category": "Rice_Grains", "subcategory": "Organic"},
            {"id": "RG3", "name": "Quinoa Imported", "base_price": 299.0, "price": 299.0, "inventory": 150, "category": "Rice_Grains", "subcategory": "Imported"},
        ],
        "Pulses": [
            {"id": "PL1", "name": "Toor Dal Premium", "base_price": 140.0, "price": 140.0, "inventory": 400, "category": "Pulses", "subcategory": "Premium"},
            {"id": "PL2", "name": "Moong Dal", "base_price": 120.0, "price": 120.0, "inventory": 350, "category": "Pulses", "subcategory": "Regular"},
            {"id": "PL3", "name": "Masoor Dal", "base_price": 110.0, "price": 110.0, "inventory": 300, "category": "Pulses", "subcategory": "Regular"},
        ]
    },
    "Dairy": {
        "Milk": [
            {"id": "DM1", "name": "Full Cream Milk", "base_price": 68.0, "price": 68.0, "inventory": 200, "category": "Milk", "subcategory": "Full_Cream"},
            {"id": "DM2", "name": "Toned Milk", "base_price": 42.0, "price": 42.0, "inventory": 300, "category": "Milk", "subcategory": "Toned"},
            {"id": "DM3", "name": "Almond Milk", "base_price": 180.0, "price": 180.0, "inventory": 100, "category": "Milk", "subcategory": "Plant_Based"},
        ],
        "Cheese": [
            {"id": "DC1", "name": "Mozzarella", "base_price": 420.0, "price": 420.0, "inventory": 50, "category": "Cheese", "subcategory": "Fresh"},
            {"id": "DC2", "name": "Processed Cheese", "base_price": 280.0, "price": 280.0, "inventory": 80, "category": "Cheese", "subcategory": "Processed"},
            {"id": "DC3", "name": "Cheese Spread", "base_price": 150.0, "price": 150.0, "inventory": 120, "category": "Cheese", "subcategory": "Spread"},
        ]
    },
    "Beverages": {
        "Soft_Drinks": [
            {"id": "BS1", "name": "Cola 2L", "base_price": 95.0, "price": 95.0, "inventory": 250, "category": "Soft_Drinks", "subcategory": "Cola"},
            {"id": "BS2", "name": "Lemon Soda 750ml", "base_price": 45.0, "price": 45.0, "inventory": 300, "category": "Soft_Drinks", "subcategory": "Soda"},
            {"id": "BS3", "name": "Orange Drink 1L", "base_price": 75.0, "price": 75.0, "inventory": 200, "category": "Soft_Drinks", "subcategory": "Fruit_Drink"},
        ],
        "Health_Drinks": [
            {"id": "BH1", "name": "Protein Shake", "base_price": 450.0, "price": 450.0, "inventory": 100, "category": "Health_Drinks", "subcategory": "Protein"},
            {"id": "BH2", "name": "Green Tea", "base_price": 280.0, "price": 280.0, "inventory": 150, "category": "Health_Drinks", "subcategory": "Tea"},
            {"id": "BH3", "name": "Energy Drink", "base_price": 99.0, "price": 99.0, "inventory": 200, "category": "Health_Drinks", "subcategory": "Energy"},
        ]
    },
    "Snacks": {
        "Chips": [
            {"id": "SC1", "name": "Potato Chips Classic", "base_price": 20.0, "price": 20.0, "inventory": 500, "category": "Chips", "subcategory": "Potato"},
            {"id": "SC2", "name": "Tortilla Chips", "base_price": 50.0, "price": 50.0, "inventory": 300, "category": "Chips", "subcategory": "Tortilla"},
            {"id": "SC3", "name": "Veggie Chips", "base_price": 40.0, "price": 40.0, "inventory": 200, "category": "Chips", "subcategory": "Veggie"},
        ],
        "Biscuits": [
            {"id": "SB1", "name": "Chocolate Cookies", "base_price": 30.0, "price": 30.0, "inventory": 400, "category": "Biscuits", "subcategory": "Cookies"},
            {"id": "SB2", "name": "Cream Biscuits", "base_price": 25.0, "price": 25.0, "inventory": 450, "category": "Biscuits", "subcategory": "Cream"},
            {"id": "SB3", "name": "Digestive", "base_price": 45.0, "price": 45.0, "inventory": 300, "category": "Biscuits", "subcategory": "Digestive"},
        ]
    },
    "Personal_Care": {
        "Soap_Body_Wash": [
            {"id": "PS1", "name": "Luxury Soap", "base_price": 45.0, "price": 45.0, "inventory": 200, "category": "Soap_Body_Wash", "subcategory": "Luxury"},
            {"id": "PS2", "name": "Body Wash", "base_price": 220.0, "price": 220.0, "inventory": 150, "category": "Soap_Body_Wash", "subcategory": "Body_Wash"},
            {"id": "PS3", "name": "Hand Wash", "base_price": 99.0, "price": 99.0, "inventory": 250, "category": "Soap_Body_Wash", "subcategory": "Hand_Wash"},
        ],
        "Hair_Care": [
            {"id": "PH1", "name": "Anti-Dandruff Shampoo", "base_price": 299.0, "price": 299.0, "inventory": 100, "category": "Hair_Care", "subcategory": "Shampoo"},
            {"id": "PH2", "name": "Hair Oil", "base_price": 150.0, "price": 150.0, "inventory": 200, "category": "Hair_Care", "subcategory": "Oil"},
            {"id": "PH3", "name": "Hair Conditioner", "base_price": 250.0, "price": 250.0, "inventory": 150, "category": "Hair_Care", "subcategory": "Conditioner"},
        ]
    }
}

def get_all_products() -> List[Dict]:
    """Get a flat list of all products with their categories"""
    products = []
    for category, subcategories in PRODUCT_CATALOG.items():
        for subcategory, items in subcategories.items():
            for item in items:
                products.append({
                    **item,
                    "category": category,
                    "subcategory": subcategory
                })
    return products

def get_product_by_id(product_id: str) -> Dict:
    """Get product details by ID"""
    for category, subcategories in PRODUCT_CATALOG.items():
        for subcategory, items in subcategories.items():
            for item in items:
                if item["id"] == product_id:
                    return {
                        **item,
                        "category": category,
                        "subcategory": subcategory
                    }
    return None
