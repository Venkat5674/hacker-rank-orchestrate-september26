import datetime
import pandas as pd

def format_date_pretty(d_str):
    d = datetime.datetime.strptime(str(d_str).strip(), "%Y-%m-%d")
    day_str = str(d.day)
    month_name = d.strftime("%B")
    year_str = d.strftime("%Y")
    return f"{day_str} {month_name} {year_str}"

def format_amt(val):
    if pd.isna(val):
        return ""
    val = float(val)
    if val.is_integer():
        return f"{int(val)}"
    return f"{val:.2f}".rstrip('0').rstrip('.')

def format_amt_commas(val):
    if pd.isna(val):
        return ""
    val = float(val)
    if val.is_integer():
        return f"{int(val):,}"
    return f"{val:,.2f}".rstrip('0').rstrip('.')

class DecisionExplainer:
    def __init__(self, data_loader):
        self.loader = data_loader

    def generate_explanation(self, request_row, res_dict, simulator):
        u_id = request_row['user_id']
        req_amt = float(request_row['requested_amount'])
        req_date_str = str(request_row['request_date'])
        deadline_str = str(request_row['desired_completion_date'])
        
        prof = self.loader.profiles_dict[u_id]
        curr = prof['home_currency']
        min_bal = float(prof['minimum_balance_to_keep'])
        
        status = res_dict['affordability_status']
        method = res_dict['recommended_payment_method']
        plan_str = res_dict['payment_plan']
        spending = res_dict['spending_changes_needed']
        
        # 1. Affordable Now (Full Payment)
        if status == 'affordable_now' and method == 'full_payment':
            if spending == 'none':
                return f"Pay {curr} {format_amt_commas(req_amt)} today. This leaves at least {curr} {format_amt_commas(min_bal)} available over the next 90 days."

        # 2. Affordable with Plan (Installments)
        if status == 'affordable_with_plan' and method == 'installments':
            parts = plan_str.split('|')
            num_pmts = len(parts)
            first_p = parts[0].split(':')
            p_date = format_date_pretty(first_p[0])
            p_amt = float(first_p[1])
            return f"Use {num_pmts} installments of {curr} {format_amt_commas(p_amt)}, starting {p_date}. This leaves at least {curr} {format_amt_commas(min_bal)} available."

        # 3. Affordable with Plan (Partial Payment)
        if status == 'affordable_with_plan' and method == 'partial_payment':
            parts = plan_str.split('|')
            p1_amt = float(parts[0].split(':')[1])
            p2_date = format_date_pretty(parts[1].split(':')[0])
            p2_amt = float(parts[1].split(':')[1])
            return f"Pay {curr} {format_amt_commas(p1_amt)} today and the remaining {curr} {format_amt_commas(p2_amt)} on {p2_date}. This completes the full request and keeps the {curr} {format_amt_commas(min_bal)} minimum protected."

        # 4. Affordable Later (Wait)
        if (status in ['affordable_later', 'affordable_with_plan']) and method == 'wait':
            wait_date_str = plan_str.split(':')[0]
            pretty_wait_date = format_date_pretty(wait_date_str)
            return f"Pay {curr} {format_amt_commas(req_amt)} in full on {pretty_wait_date}. Paying earlier would take the balance below the {curr} {format_amt_commas(min_bal)} minimum."

        # 5. Affordable with Plan + Spending Changes
        if status == 'affordable_with_plan' and spending != 'none':
            sp_parts = spending.split('|')
            sp_desc_list = []
            for sp in sp_parts:
                sp_tokens = sp.split(':')
                action = sp_tokens[0]
                ev_id = sp_tokens[1]
                ev_rows = simulator.user_events[simulator.user_events['event_id'] == ev_id]
                desc = ev_rows.iloc[0]['description'] if len(ev_rows) > 0 else ev_id
                
                if action == 'stop':
                    sp_desc_list.append(f"Stop the {desc.lower()}")
                elif action == 'reduce_to' and len(sp_tokens) >= 3:
                    new_val = float(sp_tokens[2])
                    sp_desc_list.append(f"Reduce the {desc.lower()} to {curr} {format_amt_commas(new_val)}")

            sp_text = " and ".join(sp_desc_list)
            return f"{sp_text}, then pay {curr} {format_amt_commas(req_amt)} today. This leaves at least {curr} {format_amt_commas(min_bal)} available."

        # 6. Not Affordable / Not Recommended
        if status == 'not_affordable':
            pretty_deadline = format_date_pretty(deadline_str)
            return f"Do not make this payment by {pretty_deadline}. None of the available options keeps the {curr} {format_amt_commas(min_bal)} minimum protected."

        return f"Payment of {curr} {format_amt_commas(req_amt)} is evaluated with recommendation {method} under {status}."
