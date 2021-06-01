import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pprint import pprint
import pickle

only_pos_cols = \
  ['fips', 'State_FIPS_Code', 'log_rolled_cases.x', 'shifted_time', 'log_rolled_cases.y', 'LAT', 'AREA_SQMI', 'E_TOTPOP', 'E_HU', 'E_HH', 'E_POV', 'E_UNEMP', 'E_PCI', 'E_NOHSDP', 'E_AGE65', 'E_AGE17', 'E_DISABL', 'E_SNGPNT', 'E_MINRTY', 'E_LIMENG', 'E_MUNIT', 'E_MOBILE', 'E_CROWD', 'E_NOVEH', 'E_GROUPQ', 'EP_POV', 'MP_POV', 'EP_UNEMP', 'MP_UNEMP', 'EP_PCI', 'MP_PCI', 'EP_NOHSDP', 'MP_NOHSDP', 'EP_AGE65', 'MP_AGE65', 'EP_AGE17', 'MP_AGE17', 'EP_DISABL', 'MP_DISABL', 'EP_SNGPNT', 'MP_SNGPNT', 'EP_MINRTY', 'MP_MINRTY', 'EP_LIMENG', 'MP_LIMENG', 'EP_MUNIT', 'MP_MUNIT', 'EP_MOBILE', 'MP_MOBILE', 'EP_CROWD', 'MP_CROWD', 'EP_NOVEH', 'MP_NOVEH', 'EP_GROUPQ', 'MP_GROUPQ', 'EPL_POV', 'EPL_UNEMP', 'EPL_PCI', 'EPL_NOHSDP', 'SPL_THEME1', 'RPL_THEME1', 'EPL_AGE65', 'EPL_AGE17', 'EPL_DISABL', 'EPL_SNGPNT', 'SPL_THEME2', 'RPL_THEME2', 'EPL_MINRTY', 'EPL_LIMENG', 'SPL_THEME3', 'RPL_THEME3', 'EPL_MUNIT', 'EPL_MOBILE', 'EPL_CROWD', 'EPL_NOVEH', 'EPL_GROUPQ', 'SPL_THEME4', 'RPL_THEME4', 'SPL_THEMES', 'RPL_THEMES', 'F_POV', 'F_UNEMP', 'F_PCI', 'F_NOHSDP', 'F_THEME1', 'F_AGE65', 'F_AGE17', 'F_DISABL', 'F_SNGPNT', 'F_THEME2', 'F_MINRTY', 'F_LIMENG', 'F_THEME3', 'F_MUNIT', 'F_MOBILE', 'F_CROWD', 'F_NOVEH', 'F_GROUPQ', 'F_THEME4', 'F_TOTAL', 'E_UNINSUR', 'EP_UNINSUR', 'MP_UNINSUR', 'E_DAYPOP', 'State_of_emergency', 'Date_closed_K.12_public_schools', 'Closed_day_cares', 'Reopen_day_cares', 'Date_banned_visitors_to_nursing_homes', 'Stay_at_home._shelter_in_place', 'Stay_at_home_order._issued_but_did_not_specifically_restrict_movement_of_the_general_public', 'End.relax_stay_at_home.shelter_in_place', 'Closed_other_non.essential_businesses', 'Began_to_reopen_businesses', 'Religious_Gatherings_Exempt_Without_Clear_Social_Distance_Mandate.', 'Mandate_face_mask_use_by_all_individuals_in_public_spaces', 'Second_mandate_for_facemasks_by_all_individuals_in_public_places', 'Face_mask_mandate_enforced_by_fines', 'Face_mask_mandate_enforced_by_criminal_charge.citation', 'No_legal_enforcement_of_face_mask_mandate', 'Mandate_face_mask_use_by_employees_in_public.facing_businesses', 'State_ended_statewide_mask_use_by_individuals_in_public_spaces', 'Attempt_by_state_government_to_prevent_local_governments_from_implementing_face_mask_orders', 'Alcohol.Liquor_Stores_Open', 'Allow_restaurants_to_sell_takeout_alcohol', 'Allow_restaurants_to_deliver_alcohol', 'Keep_Firearms_Sellers_Open', 'Closed_restaurants_except_take_out', 'Reopen_restaurants', 'Initially_reopen_restaurants_for_outdoor_dining_only', 'Closed_gyms', 'Reopened_gyms', 'Closed_movie_theaters', 'Reopened_movie_theaters', 'Closed_Bars', 'Reopen_bars', 'Reopened_hair_salons.barber_shops', 'Reopen_Religious_Gatherings', 'Reopen_other_non.essential_retail', 'Begin_to_Re.Close_Bars', 'Re.Close_Bars_.statewide.', 'Re.Close_Movie_Theaters_.statewide.', 'Re.close_hair_salons_and_barber_shops_.statewide.', 'Re.Close_Gyms_.statewide.', 'Re.Close_Indoor_Dining_.Statewide.', 'Re.open_indoor_dining_statewide_.x2.', 'Re.open_bars_statewide_.x2.', 'Re.open_gyms_statewide_.x2.', 'Re.open_movie_theaters_statewide_.x2.', 'Close_Bars_.x3.', 'Close_Movie_Theaters_.x3.', 'Close_Gyms_.x3.', 'Close_Indoor_Dining_.x3.', 'Mandate_quarantine_for_those_entering_the_state_from_specific_settings', 'Mandate_quarantine_for_all_individuals_entering_the_state', 'Date_all_mandated_quarantines_ended', 'Date_vaccine_allocation_plan_last_updated', 'Prioritize_older_adults_in_earlier_phase_than_essential_workers', 'Proof_of_work_eligibility_requirement_for_vaccination', 'Proof_of_age_eligibility_requirement_for_vaccination', 'Proof_of_residency_requirement_for_vaccination', 'Penalty_for_failure_to_comply_with_vaccine_distribution_requirements', 'Expanded_scope_of_practice_of_certain_health_providers_to_administer_vaccines', 'Stop_Initiation_of_Evictions_overall_or_due_to_COVID_related_issues', 'Resume_initiation_of_some_evictions', 'Stop_enforcement_of_evictions_overall_or_due_to_COVID_related_issues', 'Second_stop_of_evictions_overall_or_due_to_COVID_related_issues', 'Court_closure_led_to_Suspension_of_eviction_hearings', 'Courts_reopened_to_allow_eviction_hearings_unless_prohibited_by_other_orders', 'End_eviction_moratorium', 'Second_End_Eviction_Moratorium', 'Third_End_Eviction_Moratorium', 'Renter_grace_period_or_use_of_security_deposit_to_pay_rent', 'Order_freezing_utility_shut_offs', 'Order_freezing_gas_utility_shut_offs', 'Order_lifting_a_freeze_on_gas_utility_shut_offs', 'Order_re.implementing_freezes_on_gas_utility_shut_offs', 'Order_freezing_electric_utility_shut_offs', 'Order_lifting_a_freeze_on_electric_utility_shut_offs', 'Order_re.implementing_freezes_on_electric_utility_shut_offs', 'Order_freezing_water_utility_shut_offs', 'Order_lifting_a_freeze_on_water_utility_shut_offs', 'Order_re.implementing_freezes_on_water_utility_shut_offs', 'Order_freezing_telecom_utility_shut_offs...104', 'Order_freezing_telecom_utility_shut_offs...105', 'Order_re.implementing_freezes_on_telecom_utility_shut_offs', 'Froze_mortgage_payments', 'SNAP_Waiver.Emergency_Allotments_to_Current_SNAP_Households', 'SNAP_Waiver.Pandemic_EBT', 'SNAP_Waiver.Temporary_Suspension_of_Claims_Collection', 'X2020_Q1_SNAP_ABAWD_Time_Limit_Waiver', 'Modify_Medicaid_requirements_with_1135_waivers_.date_of_CMS_approval.', 'Reopened_ACA_enrollment_using_a_special_enrollment_period', 'State_previously_allowed_audio.only_telehealth', 'Allow_audio.only_telehealth', 'Allow.expand_Medicaid_telehealth_coverage', 'State_had_CHIP_premium_non.payment_lock.out_period_as_of_January_2019', 'Suspend_CHIP_premium_non.payment_lock.outs', 'Report_COVID.19_testing_by_race.ethnicity', 'Report_COVID.19_cases_by_race.ethnicity', 'Report_COVID.19_hospitalizations_by_race', 'Report_COVID.19_deaths_by_race.ethnicity', 'Report_COVID.19_vaccinations_by_race.ethnicity', 'Report_Indigenous_COVID.19_testing', 'Report_Indigenous_COVID.19_cases', 'Report_Indigenous_COVID.19_hospitalizations', 'Report_Indigenous_COVID.19_deaths', 'Report_Indigenous_COVID.19_vaccinations', 'Stopped_personal_visitation_in_state_prisons', 'Stopped_in.person_attorney_visits_in_state_prisons', 'Began_to_resume_visitation_in_state_prisons', 'Does_not_charge_copays_for_incarcerated_individuals', 'Waived_COVID.respiratory_illness.related_copays_during_pandemic_for_incarcerated_individuals', 'Waived_all_copays_during_pandemic_for_incarcerated_individuals', 'Did_not_waive_copays_for_incarcerated_individuals', 'Suspended_elective_medical.dental_procedures', 'Resumed_elective_medical_procedures', 'No_order_to_suspend_elective_medical_procedures_but_did_release_guidance_or_orders_to_resume.', 'Second_suspension_of_elective_medical.dental_procedures.', 'No_state_unemployment_waiting_period_prior_to_pandemic._or_date_waiting_period_waived_not_found', 'Waived_one_week_waiting_period_for_unemployment_insurance', 'Reinstated_one_week_waiting_period_for_unemployment_insurance', 'Waive_work_search_requirement_for_unemployment_insurance', 'Reinstated_work_search_requirement_for_UI', 'Expand_eligibility_of_unemployment_insurance_to_anyone_who_is_quarantined_and.or_taking_care_of_someone_who_is_quarantined', 'Expand_eligibility_to_high.risk_individuals_in_preventative_quarantine', 'Expand_eligibility_of_unemployment_insurance_to_those_who_have_lost_childcare.school_closures', 'Extend_the_amount_of_time_an_individual_can_be_on_unemployment_insurance', 'Weekly_unemployment_insurance_maximum_amount_.dollars.', 'Weekly_unemployment_insurance_maximum_amount_with_extra_stimulus_.through_July_31._2020._.dollars.', 'Unemployment_insurance_maximum_duration_.weeks.', 'Unemployment_insurance_maximum_duration_with_Pandemic_Emergency_Unemployment_Compensation_CARES_extension_.weeks.', 'Number_of_calendar_quarters_with_earnings_in_the_base_period_needed_to_qualify_for_UI.', 'Minimum_total_earnings_required_outside_the_highest_earning_calendar_quarter_of_the_base_period_to_qualify_for_UI.', 'Require_earnings_in_the_last_two_calendar_quarters_of_the_base_period_in_order_to_qualify_for_UI.', 'Earnings_in_base_period_required_to_receive_a_.300_weekly_benefit_amount_for_UI.', 'Taxable_Wage_Amount', 'Minimum_Tax_Rate_...', 'Maximum_Tax_Rate_...', 'Average_Benefit_Amount_.August.', 'Made_Effort_to_Limit_Abortion_Access', 'Use_of_telemedicine.telephone_evaluations_to_initiate_buprenorphine_prescribing', 'Patients_can_receive_14.28_take.home_doses_of_opioid_medication', 'Home_delivery_of_take.home_medication_by_opioid_treatment_programs', 'Use_of_telemedicine_for_schedule_II.V_prescriptions', 'Exceptions_to_emergency_oral_prescriptions', 'Waive_requirement_to_obtain_separate_DEA_registration_to_dispense_outside_home_state', 'Paid_sick_leave', 'Medicaid_Expansion', 'Population_density_per_square_miles', 'Population_2018', 'Square_Miles', 'Number_Homeless_.2019.', 'Percent_Unemployed_.2018.', 'Percent_living_under_the_federal_poverty_line_.2018.', 'Percent_at_risk_for_serious_illness_due_to_COVID', 'All.cause_deaths_2018', 'Mental_health_professionals_per_100.000_population_in_2019', 'Were_there_casino.s._in_State._.Land_and_Non.Land_based.', 'State_has_at_least_one_Indian.Alaska_Native_reservation', 'COVID.19_is_not_an_acceptable_reason_to_request_application_for_mail.in_ballot_unless_sick_or_exposed_.as_of_September_16._2020.', 'Witness_or_notary_signature_required_for_mail.in_ballot_.as_of_September_1._2020.', 'Permanent_mail.in_ballot_system', 'Automatic_mail.in_ballot_system_in_response_to_COVID.19_.0.1.2_2_being_conditional._see_notes_for_details.', 'Automatic_applications_sent_for_mail.in_ballots_in_response_to_COVID.19_.0.1.2_2_being_conditional._see_notes_for_details.', 'Last_date_of_receipt_of_mail.in_ballot_request_for_the_general_election_.by_mail_or_online.', 'State.Mandated_Casino_Closure', 'State.Mandated_Casino_Re.Opening.', 'Second_Casino_Closure', 'Second_Casino_Re.Opening', 'Mention_of_Tribal_Casinos', 'X2015_Minimum_Wage', 'X2016_Minimum_Wage', 'X2017_Minimum_Wage', 'X2018_Minimum_Wage', 'X2019_Minimum_Wage', 'X2020_Minimum_Wage', 'X2020_Minimum_Wage_for_Tipped_Workers', 'Different_Minimum_Wage_for_Smaller_Businesses', 'X.Planned._2021_Minimum_Wage', 'positive', 'positiveScore', 'totalTestResults', 'cutoff'] 

final_means = pickle.load(open("means.pickle", "rb"))
final_cov = pickle.load(open("covariances.pickle", "rb"))
uni_cols = pickle.load(open("unique_columns.pickle", "rb"))

print(final_means.shape)
print(final_cov.shape)

threshold = 10
# print(uni_cols.shape)
# print(uni_cols[uni_cols < threshold])
# print(uni_cols[uni_cols < threshold].shape)

# %%
samples = np.random.multivariate_normal(mean=final_means, cov=final_cov, size=100)
# print(samples.shape)
# print(samples[0])

# %%
# print(samples.shape[1] + uni_cols[uni_cols < threshold].shape[0])
# I think for everything I don't have I'll just input zero or something

# %%
# file_path = os.path.join("..", "data", "block_windowsize=2", "block_51.csv")
file_path = os.path.join("..", "data", "block_windowsize=2", "block_400.csv")
df = pd.read_csv(file_path)
# print(df.shape)
count = 0
other_cols = []
for column in df.columns:
    if column not in final_means.index and column not in uni_cols.index:
        count += 1
        # print(column)
        other_cols.append(column)
# print("Number of other columns: {}".format(count))
# print("Total we have")
# print(samples.shape[1] + uni_cols[uni_cols < threshold].shape[0] + count)
# print("Total we need: 348")

# %%
df_cols = set(list(df.columns))
listo = []
listo.extend(list(final_means.index))
listo.extend(list(uni_cols.index))
listo.extend(other_cols)
our_cols = set(listo)
# pprint(df_cols)
# pprint(our_cols)
# print(len(df_cols))
# print(len(our_cols))
# print("\nTHE FINAL THINGY")
removed_cols = {"shifted_log_rolled_cases","datetime","State_FIPS_Code","county","state","log_rolled_cases.x","shifted_time"}
# pprint(df_cols - our_cols)
# print(len(df_cols.intersection(our_cols)))
# print(len(df_cols - removed_cols))

# %%
# Okay so it seems like we have the right amount of things now, just gotta fill those in
def create_sample():
    sample = np.random.multivariate_normal(mean=final_means, cov=final_cov, size=1)
    # print(sample[0].shape)
    # print(final_means.index.shape)
    data = pd.Series(sample[0], index=final_means.index)
    # TODO randomly pick ones or 0s for some of these
    other_data = [0] * len(uni_cols[uni_cols < threshold])
    new_index = uni_cols[uni_cols < threshold].index
    other_data = pd.Series(other_data, index=new_index)
    data = data.append(other_data)
    # These might be string, but the model only takes numeric values, so...
    for col in other_cols:
        data[col] = 0

    # print(data.shape)
    # print(len(df_cols))
    excluded = df_cols - set(data.index)
    # print(excluded)

    extra = pd.Series(0, excluded)
    data = data.append(extra)
    # data[list(excluded)] = 0
    # for thing in excluded:
    #     data[thing] = 0

    data = data.reindex(df.columns)
    data = data.drop(removed_cols)
    # print(data.shape)
    fix_sample(data)
    return data

def fix_sample(sample):
    # TODO for now, lets just flip the sign of negative columns
    sample[sample.index.intersection(only_pos_cols)] = sample[sample.index.intersection(only_pos_cols)].abs()