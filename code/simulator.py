import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

def parse_date(d_str):
    if isinstance(d_str, datetime.date):
        return d_str
    return datetime.datetime.strptime(str(d_str).strip(), "%Y-%m-%d").date()

def format_date(d):
    return d.strftime("%Y-%m-%d")

class CashflowSimulator:
    def __init__(self, data_loader, user_id, request_date, forecast_days=90):
        self.loader = data_loader
        self.user_id = user_id
        self.request_date = parse_date(request_date)
        self.forecast_days = forecast_days
        self.end_date = self.request_date + timedelta(days=forecast_days)
        
        self.profile = self.loader.profiles_dict[user_id]
        self.home_currency = self.profile['home_currency']
        self.current_balance = float(self.profile['current_available_balance'])
        self.min_balance = float(self.profile['minimum_balance_to_keep'])
        
        self.user_events = self.loader.events[self.loader.events['user_id'] == user_id].copy()
        self.user_msg_updates = self.loader.user_message_updates.get(user_id, {})
        
        self.prepare_cash_flows()

    def prepare_cash_flows(self):
        """Build daily scheduled net cash flows for the forecast window [request_date, request_date + forecast_days]."""
        self.daily_net = {self.request_date + timedelta(days=i): 0.0 for i in range(self.forecast_days + 1)}
        self.event_impacts = {} # event_id -> list of impact dicts

        # 1. Existing events in dataset
        explicit_dates = set()

        for _, row in self.user_events.iterrows():
            event_id = row['event_id']
            status = str(row['status']).lower().strip()
            direction = str(row['direction']).lower().strip()
            event_type = str(row['event_type']).lower().strip()
            category = str(row['category']).lower().strip()
            
            # Rule: Exclude pending credits, unrealized investments, failed/cancelled
            if status in ['failed', 'cancelled']:
                continue
            if event_type == 'non_cash' or 'unrealized' in status or ('investment' in event_type and status != 'settled'):
                continue
            if direction == 'credit' and status not in ['settled', 'scheduled']:
                continue
            
            s_date_str = row['settlement_date'] if pd.notna(row['settlement_date']) else row['event_date']
            if pd.isna(s_date_str):
                continue
            s_date = parse_date(s_date_str)

            # Check message date override for salary
            if category == 'salary' and 'new_salary_date' in self.user_msg_updates:
                s_date = parse_date(self.user_msg_updates['new_salary_date'])

            # Ignore past events or already settled events on/before request_date
            if s_date < self.request_date:
                continue
            if s_date == self.request_date and status == 'settled':
                continue

            raw_amt = row['amount']
            if pd.isna(raw_amt):
                continue
            amt = float(raw_amt)

            # Check message amount override for salary
            if category == 'salary' and 'new_salary_amount' in self.user_msg_updates:
                amt = self.user_msg_updates['new_salary_amount']
                curr = self.user_msg_updates.get('salary_currency', row['currency'])
            else:
                curr = row['currency']

            amt_home = self.loader.convert_currency(amt, curr, self.home_currency, format_date(s_date))
            net_amt = amt_home if direction == 'credit' else -amt_home
            
            if self.request_date <= s_date <= self.end_date:
                self.daily_net[s_date] += net_amt
                explicit_dates.add((category, s_date))
                if event_id not in self.event_impacts:
                    self.event_impacts[event_id] = []
                self.event_impacts[event_id].append({
                    'date': s_date,
                    'amount': net_amt,
                    'direction': direction,
                    'category': category,
                    'raw_amount': amt_home
                })

        # 2. Recurrence Projections
        projected = self.project_recurring_events(explicit_dates)
        for proj in projected:
            dt = proj['date']
            net_amt = proj['amount']
            ev_id = proj['event_id']
            if dt in self.daily_net:
                self.daily_net[dt] += net_amt
                if ev_id not in self.event_impacts:
                    self.event_impacts[ev_id] = []
                self.event_impacts[ev_id].append({
                    'date': dt,
                    'amount': net_amt,
                    'direction': proj['direction'],
                    'category': proj['category'],
                    'raw_amount': abs(net_amt)
                })

    def project_recurring_events(self, explicit_dates):
        projected = []
        hist = self.user_events[self.user_events['status'].isin(['settled', 'scheduled'])].copy()
        hist['eff_date'] = hist['settlement_date'].fillna(hist['event_date']).apply(parse_date)
        
        groups = hist.groupby(['category', 'direction', 'event_type'])
        
        for (cat, direction, ev_type), grp in groups:
            grp = grp.sort_values('eff_date')
            if len(grp) < 2:
                continue
                
            dates = grp['eff_date'].tolist()
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_int = np.mean(intervals)
            
            # Monthly recurring (interval 25 - 35 days)
            if 25 <= avg_int <= 35:
                last_date = dates[-1]
                last_row = grp.iloc[-1]
                amt = float(last_row['amount'])
                if cat == 'salary' and 'new_salary_amount' in self.user_msg_updates:
                    amt = self.user_msg_updates['new_salary_amount']
                    curr = self.user_msg_updates.get('salary_currency', last_row['currency'])
                else:
                    curr = last_row['currency']

                amt_home = self.loader.convert_currency(amt, curr, self.home_currency, format_date(last_date))
                net_amt = amt_home if direction == 'credit' else -amt_home
                
                cur_d = last_date
                while cur_d <= self.end_date:
                    try:
                        month = cur_d.month % 12 + 1
                        year = cur_d.year + (cur_d.month // 12)
                        day = min(cur_d.day, 28)
                        cur_d = datetime.date(year, month, day)
                    except:
                        cur_d = cur_d + timedelta(days=30)

                    # Override salary date if specified in message
                    if cat == 'salary' and 'new_salary_date' in self.user_msg_updates:
                        msg_d = parse_date(self.user_msg_updates['new_salary_date'])
                        if msg_d.month == cur_d.month and msg_d.year == cur_d.year:
                            cur_d = msg_d

                    if self.request_date <= cur_d <= self.end_date:
                        if (cat, cur_d) not in explicit_dates:
                            projected.append({
                                'date': cur_d,
                                'amount': net_amt,
                                'category': cat,
                                'direction': direction,
                                'event_id': f"proj_{cat}_{cur_d}"
                            })

            # Variable periodic (interval 6 - 16 days)
            elif 6 <= avg_int <= 16:
                last_date = dates[-1]
                avg_amt = np.mean([
                    self.loader.convert_currency(float(r['amount']), r['currency'], self.home_currency, format_date(r['eff_date']))
                    for _, r in grp.iterrows()
                ])
                net_amt = avg_amt if direction == 'credit' else -avg_amt
                
                step_days = int(round(avg_int))
                cur_d = last_date + timedelta(days=step_days)
                while cur_d <= self.end_date:
                    if self.request_date <= cur_d <= self.end_date:
                        if (cat, cur_d) not in explicit_dates:
                            projected.append({
                                'date': cur_d,
                                'amount': net_amt,
                                'category': cat,
                                'direction': direction,
                                'event_id': f"proj_{cat}_{cur_d}"
                            })
                    cur_d += timedelta(days=step_days)

        return projected

    def simulate_balances(self, custom_payments=None, spending_changes=None):
        net_flows = self.daily_net.copy()
        
        # Apply spending changes
        if spending_changes:
            for change in spending_changes:
                parts = change.split(':')
                action = parts[0]
                event_id = parts[1]
                if event_id in self.event_impacts:
                    for impact in self.event_impacts[event_id]:
                        dt = impact['date']
                        old_net = impact['amount']
                        if old_net < 0:
                            if action == 'stop':
                                net_flows[dt] -= old_net
                            elif action == 'reduce_to' and len(parts) >= 3:
                                new_val = float(parts[2])
                                diff = (-old_net) - new_val
                                if diff > 0:
                                    net_flows[dt] += diff

        # Apply custom plan payments
        if custom_payments:
            for p_date, p_amt in custom_payments.items():
                p_dt = parse_date(p_date)
                if p_dt in net_flows:
                    net_flows[p_dt] -= float(p_amt)

        balances = {}
        curr = self.current_balance
        for i in range(self.forecast_days + 1):
            dt = self.request_date + timedelta(days=i)
            curr += net_flows[dt]
            balances[dt] = curr

        return balances

    def is_safe(self, custom_payments=None, spending_changes=None):
        balances = self.simulate_balances(custom_payments, spending_changes)
        min_b = min(balances.values())
        return min_b >= self.min_balance

    def get_min_balance(self, custom_payments=None, spending_changes=None):
        balances = self.simulate_balances(custom_payments, spending_changes)
        return min(balances.values())

    def compute_amount_safe_to_pay(self, spending_changes=None):
        balances_no_payment = self.simulate_balances(custom_payments=None, spending_changes=spending_changes)
        min_projected = min(balances_no_payment.values())
        safe_amt = max(0.0, min_projected - self.min_balance)
        return safe_amt

    def compute_earliest_date_for_full_payment(self, requested_amount, spending_changes=None):
        balances_no_payment = self.simulate_balances(custom_payments=None, spending_changes=spending_changes)
        dates = [self.request_date + timedelta(days=i) for i in range(self.forecast_days + 1)]
        
        for d in dates:
            before_ok = True
            for dt in dates:
                if dt < d:
                    if balances_no_payment[dt] < self.min_balance:
                        before_ok = False
                        break
                else:
                    if balances_no_payment[dt] - requested_amount < self.min_balance:
                        before_ok = False
                        break
            if before_ok:
                return format_date(d)
                
        return None
