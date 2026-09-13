import os
import pandas as pd
import numpy as np

from data_loader import DataLoader
from simulator import CashflowSimulator
from plan_ranker import PlanRanker
from explainer import DecisionExplainer

def run_evaluation():
    base_dir = r"c:\Users\pamud\Downloads\hackerrank-orchestrate-september26-main\dataset"
    loader = DataLoader(base_dir)
    ranker = PlanRanker(loader)
    explainer = DecisionExplainer(loader)

    if loader.sample_requests is None:
        print("sample_requests.csv not found!")
        return

    df_samples = loader.sample_requests
    results = []

    status_match = 0
    method_match = 0
    plan_match = 0
    safe_amt_diffs = []
    earliest_date_match = 0
    spending_changes_match = 0

    print("=================== EVALUATION ON SAMPLE REQUESTS ===================")
    for idx, req in df_samples.iterrows():
        req_id = req['request_id']
        u_id = req['user_id']
        req_date = req['request_date']
        
        sim = CashflowSimulator(loader, u_id, req_date)
        res = ranker.evaluate_request(req, sim)
        explanation = explainer.generate_explanation(req, res, sim)
        res['decision_explanation'] = explanation

        # Ground truth comparison
        gt_status = str(req['affordability_status']).strip()
        gt_method = str(req['recommended_payment_method']).strip()
        gt_plan = str(req['payment_plan']).strip()
        gt_safe = float(req['amount_safe_to_pay'])
        gt_date = str(req['earliest_date_for_full_payment']).strip() if pd.notna(req['earliest_date_for_full_payment']) else ""
        gt_spending = str(req['spending_changes_needed']).strip()

        pred_status = res['affordability_status']
        pred_method = res['recommended_payment_method']
        pred_plan = res['payment_plan']
        pred_safe = float(res['amount_safe_to_pay'])
        pred_date = str(res['earliest_date_for_full_payment']).strip()
        pred_spending = res['spending_changes_needed']

        # Check matches
        m_status = pred_status == gt_status
        m_method = pred_method == gt_method
        m_plan = pred_plan == gt_plan
        diff_safe = abs(pred_safe - gt_safe)
        m_date = pred_date == gt_date
        m_spending = pred_spending == gt_spending

        if m_status: status_match += 1
        if m_method: method_match += 1
        if m_plan: plan_match += 1
        if m_date: earliest_date_match += 1
        if m_spending: spending_changes_match += 1
        safe_amt_diffs.append(diff_safe)

        print(f"[{req_id}] Status: {pred_status} (GT: {gt_status}) | Match: {m_status}")
        print(f"       Method: {pred_method} (GT: {gt_method}) | Match: {m_method}")
        print(f"       SafeToday: {pred_safe} (GT: {gt_safe}) | Diff: {diff_safe:.2f}")
        print(f"       Plan: {pred_plan}")
        print(f"       GT Plan: {gt_plan}")
        print(f"       EarliestDate: '{pred_date}' (GT: '{gt_date}') | Match: {m_date}")
        print(f"       Spending: {pred_spending} (GT: {gt_spending}) | Match: {m_spending}")
        print(f"       Expl: {explanation}")
        print("-" * 75)

    n = len(df_samples)
    print("=================== SUMMARY METRICS ===================")
    print(f"Status Accuracy:         {status_match}/{n} ({status_match/n*100:.1f}%)")
    print(f"Method Accuracy:         {method_match}/{n} ({method_match/n*100:.1f}%)")
    print(f"Plan Match:              {plan_match}/{n} ({plan_match/n*100:.1f}%)")
    print(f"Earliest Date Match:     {earliest_date_match}/{n} ({earliest_date_match/n*100:.1f}%)")
    print(f"Spending Changes Match:  {spending_changes_match}/{n} ({spending_changes_match/n*100:.1f}%)")
    print(f"Mean Safe Amount Diff:   {np.mean(safe_amt_diffs):.2f}")

if __name__ == "__main__":
    run_evaluation()
