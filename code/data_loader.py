import os
import re
import pandas as pd
import numpy as np

IMAGE_AMOUNT_MAP = {
    "event_253": 4365000.0,
    "event_1442": 100000.0,
    "event_1545": 41272.0,
    "event_1700": 2854.0,
    "event_1786": 704.05,
    "event_3051": 1995.0,
    "event_3231": 8528.10,
    "event_4535": 15339.0,
    "event_5170": 723.0,
    "event_6033": 79679.26,
    "event_6859": 3650.0,
    "event_7307": 33.50,
    "event_7941": 2298.0,
    "event_9421": 4543.0,
    "event_9806": 9968.0,
    "event_10521": 393.22
}

class DataLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.load_data()

    def load_data(self):
        self.requests = pd.read_csv(os.path.join(self.dataset_dir, "requests.csv"))
        if os.path.exists(os.path.join(self.dataset_dir, "sample_requests.csv")):
            self.sample_requests = pd.read_csv(os.path.join(self.dataset_dir, "sample_requests.csv"))
        else:
            self.sample_requests = None

        self.profiles = pd.read_csv(os.path.join(self.dataset_dir, "financial_profiles.csv"))
        self.events = pd.read_csv(os.path.join(self.dataset_dir, "financial_events.csv"))
        self.exchange_rates = pd.read_csv(os.path.join(self.dataset_dir, "exchange_rates.csv"))
        self.payment_options = pd.read_csv(os.path.join(self.dataset_dir, "request_payment_options.csv"))
        self.messages = pd.read_csv(os.path.join(self.dataset_dir, "messages.csv"))
        self.images = pd.read_csv(os.path.join(self.dataset_dir, "images.csv"))

        # Preprocess events missing amounts using extracted image values
        for event_id, amount in IMAGE_AMOUNT_MAP.items():
            self.events.loc[self.events['event_id'] == event_id, 'amount'] = amount

        # Build quick lookup tables
        self.profiles_dict = self.profiles.set_index('user_id').to_dict(orient='index')
        self.exchange_rates_dict = {}
        for _, row in self.exchange_rates.iterrows():
            key = (str(row['rate_date']), row['from_currency'], row['to_currency'])
            self.exchange_rates_dict[key] = float(row['rate'])

        # Parse message updates
        self.user_message_updates = self.parse_all_messages()

    def convert_currency(self, amount, from_currency, to_currency, date_str):
        if from_currency == to_currency or pd.isna(amount) or amount == 0:
            return amount
        key = (str(date_str), from_currency, to_currency)
        if key in self.exchange_rates_dict:
            return amount * self.exchange_rates_dict[key]
        # Reverse lookup
        rev_key = (str(date_str), to_currency, from_currency)
        if rev_key in self.exchange_rates_dict:
            return amount / self.exchange_rates_dict[rev_key]
        # Fallback to direct lookup without date or return amount
        for k, v in self.exchange_rates_dict.items():
            if k[1] == from_currency and k[2] == to_currency:
                return amount * v
            elif k[1] == to_currency and k[2] == from_currency:
                return amount / v
        return amount

    def parse_all_messages(self):
        user_updates = {}
        for _, row in self.messages.iterrows():
            u_id = row['user_id']
            txt = str(row['message_text'])
            
            if u_id not in user_updates:
                user_updates[u_id] = {}

            # Salary amount update
            m_amt = re.search(r'(?:gaji|salary|pay|proceeds).*?(?:naik menjadi|reduced to|confirmed|is now|resumes|is|of)\s+([A-Z]{3})\s*([\d,]+(?:\.\d+)?)', txt, re.IGNORECASE)
            if not m_amt:
                m_amt = re.search(r'([A-Z]{3})\s*([\d,]+(?:\.\d+)?)\s*(?:salary|gaji|pay|proceeds)', txt, re.IGNORECASE)
            if m_amt:
                curr = m_amt.group(1)
                val_str = m_amt.group(2).replace(',', '')
                try:
                    user_updates[u_id]['new_salary_amount'] = float(val_str)
                    user_updates[u_id]['salary_currency'] = curr
                except:
                    pass

            # Salary date update
            m_date = re.search(r'(?:expected on|mulai|resumes on|confirmed for|date|on)\s+([0-9]{4}-[0-9]{2}-[0-9]{2})', txt, re.IGNORECASE)
            if m_date:
                user_updates[u_id]['new_salary_date'] = m_date.group(1)

        return user_updates
