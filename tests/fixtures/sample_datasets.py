"""Sample datasets for testing LLM Tab Cleaner functionality."""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def create_customer_data_sample():
    """Create a realistic customer dataset with various data quality issues."""
    np.random.seed(42)  # For reproducible tests
    
    size = 1000
    
    # Generate base data
    customer_ids = list(range(1, size + 1))
    
    # Add some duplicate IDs
    duplicate_indices = np.random.choice(len(customer_ids), size=50, replace=False)
    for idx in duplicate_indices[:25]:
        customer_ids[idx] = customer_ids[idx + 1] if idx + 1 < len(customer_ids) else customer_ids[idx - 1]
    
    # Names with various case and formatting issues
    first_names = [
        'john', 'JANE', 'Bob', 'alice', 'CHARLIE', 'diana', 'EDWARD', 'fiona',
        'george', 'HELEN', 'ian', 'JULIA', 'kevin', 'LAURA', 'mike', 'nina'
    ] * (size // 16 + 1)
    first_names = first_names[:size]
    
    last_names = [
        'smith', 'DOE', 'Johnson', 'BROWN', 'wilson', 'MILLER', 'davis', 'GARCIA',
        'rodriguez', 'MARTINEZ', 'hernandez', 'LOPEZ', 'gonzalez', 'WILSON', 'anderson', 'THOMAS'
    ] * (size // 16 + 1)
    last_names = last_names[:size]
    
    # Email addresses with various issues
    emails = []
    for i, (first, last) in enumerate(zip(first_names, last_names)):
        if i % 10 == 0:  # 10% invalid emails
            if i % 3 == 0:
                emails.append(f"{first.lower()}.{last.lower()}@invalid")  # Missing TLD
            elif i % 3 == 1:
                emails.append(f"{first.lower()}{last.lower()}@test")  # Missing .com
            else:
                emails.append(f"{first.lower()}@")  # Incomplete
        elif i % 15 == 0:  # Some with formatting issues
            emails.append(f"{first.upper()}.{last.upper()}@TEST.COM")
        else:
            domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'company.com', 'test.org']
            domain = np.random.choice(domains)
            emails.append(f"{first.lower()}.{last.lower()}@{domain}")
    
    # Phone numbers with various formats
    phone_formats = [
        lambda: f"({np.random.randint(200, 999)}) {np.random.randint(200, 999)}-{np.random.randint(1000, 9999)}",
        lambda: f"{np.random.randint(200, 999)}.{np.random.randint(200, 999)}.{np.random.randint(1000, 9999)}",
        lambda: f"{np.random.randint(200, 999)}-{np.random.randint(200, 999)}-{np.random.randint(1000, 9999)}",
        lambda: f"{np.random.randint(200, 999)}{np.random.randint(200, 999)}{np.random.randint(1000, 9999)}",
    ]
    
    phones = []
    for i in range(size):
        if i % 20 == 0:  # 5% invalid phones
            phones.append("invalid-phone" if i % 2 == 0 else None)
        else:
            format_func = np.random.choice(phone_formats)
            phones.append(format_func())
    
    # Birth dates with various formats
    birth_dates = []
    date_formats = [
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%m/%d/%Y"),
        lambda d: d.strftime("%B %d, %Y"),
        lambda d: d.strftime("%d-%m-%Y"),
    ]
    
    for i in range(size):
        if i % 25 == 0:  # 4% invalid dates
            birth_dates.append("invalid-date" if i % 2 == 0 else None)
        else:
            # Generate random birth date between 1950 and 2005
            start_date = datetime(1950, 1, 1)
            end_date = datetime(2005, 12, 31)
            random_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days)
            )
            format_func = np.random.choice(date_formats)
            birth_dates.append(format_func(random_date))
    
    # Income with various formats
    incomes = []
    for i in range(size):
        if i % 30 == 0:  # Some invalid incomes
            incomes.append("confidential" if i % 2 == 0 else None)
        elif i % 15 == 0:  # Some with currency formatting
            amount = np.random.randint(30000, 150000)
            if i % 3 == 0:
                incomes.append(f"${amount:,}")
            elif i % 3 == 1:
                incomes.append(f"{amount // 1000}k")
            else:
                incomes.append(f"${amount // 1000}K")
        else:
            incomes.append(np.random.randint(30000, 150000))
    
    # States with various formats
    states_data = [
        ('California', 'CA'), ('Texas', 'TX'), ('Florida', 'FL'), ('New York', 'NY'),
        ('Pennsylvania', 'PA'), ('Illinois', 'IL'), ('Ohio', 'OH'), ('Georgia', 'GA'),
        ('North Carolina', 'NC'), ('Michigan', 'MI'), ('New Jersey', 'NJ'), ('Virginia', 'VA'),
        ('Washington', 'WA'), ('Arizona', 'AZ'), ('Massachusetts', 'MA'), ('Tennessee', 'TN')
    ]
    
    states = []
    for i in range(size):
        if i % 40 == 0:  # Some invalid states
            states.append("Unknown" if i % 2 == 0 else None)
        else:
            state_full, state_abbrev = np.random.choice(states_data)
            if i % 10 == 0:  # Mix of full names and abbreviations
                states.append(state_full)
            elif i % 10 == 1:
                states.append(state_abbrev.lower())
            else:
                states.append(state_abbrev)
    
    return pd.DataFrame({
        'customer_id': customer_ids,
        'first_name': first_names,
        'last_name': last_names,
        'email': emails,
        'phone': phones,
        'birth_date': birth_dates,
        'annual_income': incomes,
        'state': states
    })


def create_financial_transactions_sample():
    """Create a financial transactions dataset with data quality issues."""
    np.random.seed(123)
    
    size = 2000
    
    # Transaction IDs (some duplicates)
    transaction_ids = [f"TXN{i:06d}" for i in range(1, size + 1)]
    duplicate_indices = np.random.choice(len(transaction_ids), size=30, replace=False)
    for idx in duplicate_indices[:15]:
        transaction_ids[idx] = transaction_ids[idx + 1] if idx + 1 < len(transaction_ids) else transaction_ids[idx - 1]
    
    # Dates with various formats
    start_date = datetime(2023, 1, 1)
    dates = []
    date_formats = [
        lambda d: d.strftime("%Y-%m-%d"),
        lambda d: d.strftime("%m/%d/%Y"),
        lambda d: d.strftime("%d-M-%Y"),
        lambda d: d.strftime("%Y%m%d"),
    ]
    
    for i in range(size):
        if i % 50 == 0:  # Some invalid dates
            dates.append("2023-13-45" if i % 2 == 0 else None)
        else:
            random_date = start_date + timedelta(days=np.random.randint(0, 365))
            format_func = np.random.choice(date_formats)
            dates.append(format_func(random_date))
    
    # Amounts with various formats
    amounts = []
    for i in range(size):
        if i % 100 == 0:  # Some invalid amounts
            amounts.append("PENDING" if i % 2 == 0 else None)
        else:
            base_amount = np.random.uniform(10.0, 10000.0)
            if i % 20 == 0:  # Currency formatting
                amounts.append(f"${base_amount:,.2f}")
            elif i % 20 == 1:  # Negative amounts
                amounts.append(f"-{base_amount:.2f}")
            elif i % 20 == 2:  # No decimal
                amounts.append(f"{int(base_amount)}")
            else:
                amounts.append(f"{base_amount:.2f}")
    
    # Categories with inconsistent naming
    categories = []
    category_variations = {
        'food': ['food', 'Food', 'FOOD', 'dining', 'restaurant'],
        'transport': ['transport', 'Transport', 'TRANSPORT', 'travel', 'gas'],
        'shopping': ['shopping', 'Shopping', 'SHOPPING', 'retail', 'store'],
        'utilities': ['utilities', 'Utilities', 'UTILITIES', 'bills', 'electric'],
        'entertainment': ['entertainment', 'Entertainment', 'fun', 'movies', 'games']
    }
    
    for i in range(size):
        if i % 60 == 0:  # Some invalid categories
            categories.append("OTHER" if i % 2 == 0 else None)
        else:
            category_group = np.random.choice(list(category_variations.keys()))
            categories.append(np.random.choice(category_variations[category_group]))
    
    # Merchant names with various formatting issues
    merchants = [
        'amazon', 'WALMART', 'Target', 'starbucks', 'MCDONALDS', 'Shell Gas',
        'HOME DEPOT', 'best buy', 'COSTCO', 'kroger', 'CVS PHARMACY', 'walgreens'
    ] * (size // 12 + 1)
    merchants = merchants[:size]
    
    # Add some formatting issues
    for i in range(len(merchants)):
        if i % 30 == 0:  # Add extra spaces or punctuation
            merchants[i] = f"  {merchants[i]}  "
        elif i % 30 == 1:
            merchants[i] = f"{merchants[i]}***"
        elif i % 30 == 2:
            merchants[i] = merchants[i].replace(' ', '_')
    
    return pd.DataFrame({
        'transaction_id': transaction_ids,
        'date': dates,
        'amount': amounts,
        'category': categories,
        'merchant': merchants,
        'account_type': np.random.choice(['checking', 'savings', 'credit'], size=size)
    })


def create_product_catalog_sample():
    """Create a product catalog with data quality issues."""
    np.random.seed(456)
    
    size = 1500
    
    # Product names with case and formatting issues
    product_prefixes = ['Smart', 'Digital', 'Wireless', 'Portable', 'Professional', 'Premium']
    product_types = ['Phone', 'Laptop', 'Camera', 'Speaker', 'Watch', 'Tablet']
    product_suffixes = ['Pro', 'Max', 'Plus', 'Mini', 'Ultra', 'Lite']
    
    product_names = []
    for i in range(size):
        if i % 80 == 0:  # Some invalid names
            product_names.append("" if i % 2 == 0 else None)
        else:
            prefix = np.random.choice(product_prefixes)
            ptype = np.random.choice(product_types)
            suffix = np.random.choice(product_suffixes)
            
            name = f"{prefix} {ptype} {suffix}"
            if i % 15 == 0:
                name = name.upper()
            elif i % 15 == 1:
                name = name.lower()
            
            product_names.append(name)
    
    # Prices with various formats
    prices = []
    for i in range(size):
        if i % 70 == 0:  # Some invalid prices
            prices.append("CALL FOR PRICE" if i % 2 == 0 else None)
        else:
            base_price = np.random.uniform(50.0, 2000.0)
            if i % 10 == 0:
                prices.append(f"${base_price:.2f}")
            elif i % 10 == 1:
                prices.append(f"USD {base_price:.2f}")
            elif i % 10 == 2:
                prices.append(f"{base_price:.0f}")
            else:
                prices.append(f"{base_price:.2f}")
    
    # SKUs with formatting variations
    skus = []
    for i in range(size):
        if i % 90 == 0:  # Some missing SKUs
            skus.append(None)
        else:
            sku = f"SKU{i:06d}"
            if i % 20 == 0:
                sku = sku.lower()
            elif i % 20 == 1:
                sku = f"sku-{i:06d}"
            skus.append(sku)
    
    # Categories with hierarchical inconsistencies
    categories = []
    category_hierarchy = {
        'Electronics': ['Electronics', 'ELECTRONICS', 'electronics', 'Tech'],
        'Clothing': ['Clothing', 'CLOTHING', 'clothing', 'Apparel'],
        'Home': ['Home', 'HOME', 'home', 'House', 'Household'],
        'Sports': ['Sports', 'SPORTS', 'sports', 'Fitness', 'Athletic']
    }
    
    for i in range(size):
        if i % 50 == 0:  # Some invalid categories
            categories.append("Misc" if i % 2 == 0 else None)
        else:
            main_cat = np.random.choice(list(category_hierarchy.keys()))
            categories.append(np.random.choice(category_hierarchy[main_cat]))
    
    return pd.DataFrame({
        'product_id': range(1, size + 1),
        'product_name': product_names,
        'price': prices,
        'sku': skus,
        'category': categories,
        'in_stock': np.random.choice([True, False, 'Yes', 'No', 1, 0, None], size=size),
        'weight': [f"{np.random.uniform(0.1, 10.0):.1f}lbs" if i % 30 != 0 else None for i in range(size)]
    })


def create_healthcare_sample():
    """Create a healthcare dataset with sensitive data and quality issues."""
    np.random.seed(789)
    
    size = 800
    
    # Patient IDs
    patient_ids = [f"P{i:06d}" for i in range(1, size + 1)]
    
    # Names with various privacy levels
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'REDACTED', '***'] * (size // 8 + 1)
    first_names = first_names[:size]
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'REDACTED', '***', None] * (size // 8 + 1)
    last_names = last_names[:size]
    
    # Ages with various formats and validation issues
    ages = []
    for i in range(size):
        if i % 50 == 0:  # Some invalid ages
            ages.append(-5 if i % 2 == 0 else 200)
        elif i % 25 == 0:  # Some text ages
            ages.append("elderly" if i % 2 == 0 else "infant")
        else:
            ages.append(np.random.randint(0, 100))
    
    # Diagnoses with coding inconsistencies
    diagnoses = [
        'Hypertension', 'DIABETES', 'diabetes type 2', 'High Blood Pressure',
        'Heart Disease', 'HEART_DISEASE', 'Cancer', 'CANCER', 'Flu', 'Common Cold'
    ] * (size // 10 + 1)
    diagnoses = diagnoses[:size]
    
    # Treatment dates
    treatment_dates = []
    for i in range(size):
        if i % 40 == 0:  # Some invalid dates
            treatment_dates.append("ongoing" if i % 2 == 0 else None)
        else:
            date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            treatment_dates.append(date.strftime("%Y-%m-%d"))
    
    return pd.DataFrame({
        'patient_id': patient_ids,
        'first_name': first_names,
        'last_name': last_names,
        'age': ages,
        'diagnosis': diagnoses,
        'treatment_date': treatment_dates,
        'insurance_claim_amount': [np.random.uniform(100, 50000) if i % 30 != 0 else None for i in range(size)]
    })


# Export functions for easy import
__all__ = [
    'create_customer_data_sample',
    'create_financial_transactions_sample', 
    'create_product_catalog_sample',
    'create_healthcare_sample'
]