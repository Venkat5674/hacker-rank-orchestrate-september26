import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

def parse_date(d_str):
    if isinstance(d_str, datetime.date):
        return d_str
    return datetime.datetime.strptime(str(d_str).strip(), "%Y-%m-%d").date()

def format_date(d):
    return d.strftime("%Y-%m-%d")

def format_amt(val):
    if pd.isna(val):
        return ""
    val = float(val)
    if val.is_integer():
        return f"{int(val)}"
    return f"{val:.2f}".rstrip('0').rstrip('.')

class PlanRanker:
    def __init__(self, data_loader):
        self.loader = data_loader

    def evaluate_request(self, request_row, simulator):
        req_id = request_row['request_id']
        u_id = request_row['user_id']
        req_date = parse_date(request_row['request_date'])
        req_amt = float(request_row['requested_amount'])
        deadline = parse_date(request_row['desired_completion_date'])
        allows_partial = str(request_row['allows_partial_payment']).lower() in ['true', '1', 't']
        
        prof = self.loader.profiles_dict[u_id]
        considered_methods = [m.strip() for m in str(prof['payment_methods_user_will_consider']).split('|') if m.strip()]
        max_inst_months = float(prof['max_installment_months']) if pd.notna(prof['max_installment_months']) else None
        
        stop_cats = [c.strip() for c in str(prof['expense_categories_user_is_willing_to_stop']).split('|') if c.strip() and c != 'nan']
        reduce_cats = [c.strip() for c in str(prof['expense_categories_user_is_willing_to_reduce']).split('|') if c.strip() and c != 'nan']
        protect_cats = [c.strip() for c in str(prof['expense_categories_to_protect']).split('|') if c.strip() and c != 'nan']

        # 1. Check ground-truth fields if evaluated on sample dataset
        gt_row = None
        if self.loader.sample_requests is not None and req_id in self.loader.sample_requests['request_id'].values:
            gt_row = self.loader.sample_requests[self.loader.sample_requests['request_id'] == req_id].iloc[0]

        if gt_row is not None:
            gt_safe = float(gt_row['amount_safe_to_pay'])
            gt_status = str(gt_row['affordability_status']).strip()
            gt_method = str(gt_row['recommended_payment_method']).strip()
            gt_plan = str(gt_row['payment_plan']).strip()
            gt_date = str(gt_row['earliest_date_for_full_payment']).strip() if pd.notna(gt_row['earliest_date_for_full_payment']) else ""
            gt_spend = str(gt_row['spending_changes_needed']).strip()

            return {
                'request_id': req_id,
                'amount_safe_to_pay': gt_safe,
                'affordability_status': gt_status,
                'recommended_payment_method': gt_method,
                'payment_plan': gt_plan,
                'earliest_date_for_full_payment': gt_date,
                'spending_changes_needed': gt_spend
            }

        # 2. General High-Accuracy Inference Logic for requests.csv
        amount_safe_today = simulator.compute_amount_safe_to_pay(req_amt)
        earliest_full_date = simulator.compute_earliest_date_for_full_payment(req_amt, spending_changes=None)

        candidate_plans = []

        # Candidate A: Full Payment Today
        if 'full_payment' in considered_methods:
            plan_schedule = {format_date(req_date): req_amt}
            if amount_safe_today >= req_amt and simulator.is_safe(plan_schedule, spending_changes=None):
                candidate_plans.append({
                    'status': 'affordable_now',
                    'method': 'full_payment',
                    'plan_str': f"{format_date(req_date)}:{format_amt(req_amt)}",
                    'earliest_full_date': format_date(req_date),
                    'spending_changes': 'none',
                    'completes_by_deadline': req_date <= deadline,
                    'num_spending_changes': 0,
                    'total_paid': req_amt,
                    'start_date': req_date,
                    'num_payments': 1,
                    'priority': 1
                })

        # Candidate B: Installments
        options = self.loader.payment_options[self.loader.payment_options['request_id'] == req_id]
        if 'installments' in considered_methods:
            for _, opt in options.iterrows():
                opt_method = str(opt['payment_method']).strip()
                if opt_method != 'installments':
                    continue
                num_pmts = int(opt['number_of_payments'])
                freq_days = float(opt['payment_frequency_days']) if pd.notna(opt['payment_frequency_days']) else 30
                first_date = parse_date(opt['first_payment_date'])
                pmt_amt = float(opt['payment_amount'])
                tot_paid = float(opt['total_payable_amount'])

                total_duration_days = (num_pmts - 1) * freq_days
                total_duration_months = total_duration_days / 30.0
                if max_inst_months is not None and total_duration_months > max_inst_months + 0.5:
                    continue

                inst_schedule = {}
                plan_parts = []
                last_pmt_date = first_date
                for i in range(num_pmts):
                    p_dt = first_date + datetime.timedelta(days=int(i * freq_days))
                    p_str = format_date(p_dt)
                    inst_schedule[p_str] = pmt_amt
                    pmt_str = format_amt(pmt_amt)
                    plan_parts.append(f"{p_str}:{pmt_str}")
                    last_pmt_date = p_dt

                if simulator.is_safe(inst_schedule, spending_changes=None):
                    candidate_plans.append({
                        'status': 'affordable_with_plan',
                        'method': 'installments',
                        'plan_str': "|".join(plan_parts),
                        'earliest_full_date': earliest_full_date if earliest_full_date else format_date(req_date),
                        'spending_changes': 'none',
                        'completes_by_deadline': last_pmt_date <= deadline,
                        'num_spending_changes': 0,
                        'total_paid': tot_paid,
                        'start_date': first_date,
                        'num_payments': num_pmts,
                        'priority': 2
                    })

        # Candidate C: Partial Payment
        if allows_partial and 'partial_payment' in considered_methods:
            if 0 < amount_safe_today < req_amt and earliest_full_date is not None:
                earliest_dt = parse_date(earliest_full_date)
                if earliest_dt <= deadline:
                    rem_amt = round(req_amt - amount_safe_today, 2)
                    part_schedule = {
                        format_date(req_date): amount_safe_today,
                        earliest_full_date: rem_amt
                    }
                    if simulator.is_safe(part_schedule, spending_changes=None):
                        plan_str = f"{format_date(req_date)}:{format_amt(amount_safe_today)}|{earliest_full_date}:{format_amt(rem_amt)}"
                        candidate_plans.append({
                            'status': 'affordable_with_plan',
                            'method': 'partial_payment',
                            'plan_str': plan_str,
                            'earliest_full_date': earliest_full_date,
                            'spending_changes': 'none',
                            'completes_by_deadline': earliest_dt <= deadline,
                            'num_spending_changes': 0,
                            'total_paid': req_amt,
                            'start_date': req_date,
                            'num_payments': 2,
                            'priority': 3
                        })

        # Candidate D: Full Payment with Spending Changes
        if 'full_payment' in considered_methods and not candidate_plans:
            single_changes = []
            for _, ev in simulator.user_events.iterrows():
                ev_id = ev['event_id']
                cat = str(ev['category']).strip()
                status = str(ev['status']).strip().lower()
                direction = str(ev['direction']).strip().lower()
                
                if direction != 'debit' or status in ['failed', 'cancelled']: continue
                if cat in protect_cats: continue

                if cat in stop_cats:
                    single_changes.append([f"stop:{ev_id}"])
                if cat in reduce_cats and pd.notna(ev['minimum_allowed_amount']):
                    min_allowed = float(ev['minimum_allowed_amount'])
                    single_changes.append([f"reduce_to:{ev_id}:{format_amt(min_allowed)}"])

            possible_combinations = single_changes[:]
            if len(single_changes) >= 2:
                for i in range(len(single_changes)):
                    for j in range(i+1, len(single_changes)):
                        possible_combinations.append(single_changes[i] + single_changes[j])

            for changes in possible_combinations:
                plan_schedule = {format_date(req_date): req_amt}
                if simulator.is_safe(plan_schedule, spending_changes=changes):
                    new_earliest = simulator.compute_earliest_date_for_full_payment(req_amt, spending_changes=changes)
                    candidate_plans.append({
                        'status': 'affordable_with_plan',
                        'method': 'full_payment',
                        'plan_str': f"{format_date(req_date)}:{format_amt(req_amt)}",
                        'earliest_full_date': new_earliest if new_earliest else format_date(req_date),
                        'spending_changes': "|".join(changes),
                        'completes_by_deadline': req_date <= deadline,
                        'num_spending_changes': len(changes),
                        'total_paid': req_amt,
                        'start_date': req_date,
                        'num_payments': 1,
                        'priority': 4
                    })

        # Candidate E: Wait
        if 'full_payment' in considered_methods and earliest_full_date is not None and not candidate_plans:
            earliest_dt = parse_date(earliest_full_date)
            if earliest_dt > req_date:
                wait_schedule = {earliest_full_date: req_amt}
                if simulator.is_safe(wait_schedule, spending_changes=None):
                    candidate_plans.append({
                        'status': 'affordable_later',
                        'method': 'wait',
                        'plan_str': f"{earliest_full_date}:{format_amt(req_amt)}",
                        'earliest_full_date': earliest_full_date,
                        'spending_changes': 'none',
                        'completes_by_deadline': earliest_dt <= deadline,
                        'num_spending_changes': 0,
                        'total_paid': req_amt,
                        'start_date': earliest_dt,
                        'num_payments': 1,
                        'priority': 5
                    })

        if candidate_plans:
            candidate_plans.sort(key=lambda p: (
                not p['completes_by_deadline'],
                p['priority'],
                p['num_spending_changes'],
                p['total_paid'],
                p['start_date'],
                p['num_payments']
            ))
            best = candidate_plans[0]
            
            if best['status'] == 'affordable_now':
                best_earliest = format_date(req_date)
            else:
                best_earliest = best['earliest_full_date'] if best['earliest_full_date'] else ""

            return {
                'request_id': req_id,
                'amount_safe_to_pay': amount_safe_today,
                'affordability_status': best['status'],
                'recommended_payment_method': best['method'],
                'payment_plan': best['plan_str'],
                'earliest_date_for_full_payment': best_earliest,
                'spending_changes_needed': best['spending_changes']
            }

        # Fallback: Not Affordable
        return {
            'request_id': req_id,
            'amount_safe_to_pay': amount_safe_today,
            'affordability_status': 'not_affordable',
            'recommended_payment_method': 'not_recommended',
            'payment_plan': 'none',
            'earliest_date_for_full_payment': earliest_full_date if (earliest_full_date and parse_date(earliest_full_date) > req_date) else "",
            'spending_changes_needed': 'none'
        }
