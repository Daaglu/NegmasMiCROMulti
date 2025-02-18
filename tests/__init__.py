import itertools
import numpy as np
from negmas.gb.negotiators.timebased import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)
from negmas import make_issue, SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
from src.negmas.gb.negotiators.micro_multi import MiCRONegotiatorMulti
from itertools import permutations
from negmas.gb.negotiators.timebased import BoulwareTBNegotiator, ConcederTBNegotiator, LinearTBNegotiator

#from negmas.gb.negotiators import MiCRONegotiator
from src.negmas.gb.negotiators.micro import MiCRONegotiator

from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun
from negmas.preferences.value_fun import LinearFun, IdentityFun, AffineFun
import numpy as np
from pathlib import Path
from negmas.inout import Scenario
from negmas.sao.negotiators import AspirationNegotiator
from negmas.genius.gnegotiators import Atlas3, PonPokoAgent, CaduceusDC16, KakeSoba, AgreeableAgent2018
import statistics
import math
import xml.etree.ElementTree as ET

def create_utility_functions(outcome_space):
    return [
        LUFun(values={"price": AffineFun(-1.5, bias=15.0), "quantity": LinearFun(0.3),
                      "delivery_time": AffineFun(-2.0, bias=20.0), "warranty_period": LinearFun(0.4),
                      "quality": LinearFun(0.8)},
              weights={"price": 3.0, "quantity": 1.0, "delivery_time": 5.0, "warranty_period": 2.0, "quality": 4.0},
              outcome_space=outcome_space).scale_max(1.0),
        LUFun(values={"price": AffineFun(-0.5, bias=8.0), "quantity": LinearFun(0.6),
                      "delivery_time": AffineFun(-1.0, bias=12.0), "warranty_period": LinearFun(1.5),
                      "quality": AffineFun(1.0, bias=5.0)},
              weights={"price": 2.0, "quantity": 3.0, "delivery_time": 2.0, "warranty_period": 4.0, "quality": 3.0},
              outcome_space=outcome_space).scale_max(1.0),
        LUFun(values={"price": AffineFun(-2.0, bias=20.0), "quantity": LinearFun(0.1), "delivery_time": IdentityFun(),
                      "warranty_period": AffineFun(-0.8, bias=10.0), "quality": LinearFun(2.0)},
              weights={"price": 5.0, "quantity": 1.0, "delivery_time": 2.0, "warranty_period": 1.5, "quality": 5.0},
              outcome_space=outcome_space).scale_max(1.0),
        LUFun(values={"price": IdentityFun(), "quantity": AffineFun(1.0, bias=0.0),
                      "delivery_time": AffineFun(-0.5, bias=5.0), "warranty_period": LinearFun(0.2),
                      "quality": LinearFun(0.9)},
              weights={"price": 1.0, "quantity": 2.5, "delivery_time": 1.0, "warranty_period": 0.5, "quality": 3.0},
              outcome_space=outcome_space).scale_max(1.0),
    ]

def run_test():
    issues = [
        make_issue(name="price", values=(1, 10)),
        make_issue(name="quantity", values=(1, 10)),
        make_issue(name="delivery_time", values=(1, 10)),
        make_issue(name="warranty_period", values=(1, 10)),
        make_issue(name="quality", values=(1, 20)),
    ]

    session = SAOMechanism(issues=issues, time_limit=35)
    utilities = create_utility_functions(session.outcome_space)

    agent1 = MiCRONegotiator(preferences=utilities[0].scale_max(1.0),name="Micro")
    agent2 = AgreeableAgent2018(preferences=utilities[2].scale_max(1.0), name="Agreeable")
    agent3 = KakeSoba(preferences=utilities[1].scale_max(1.0), name="KakeSoba")
    session.add(agent1)
    session.add(agent2)
    session.add(agent3)

    print(session.run())
    proposer = session.state.current_proposer
    print(session.extended_trace)
    if session.agreement is not None:
        print("Final Agreement:", session.agreement)
        for i, negotiator in enumerate(session.negotiators):
            role = "Proposer" if negotiator.id == proposer else "Responder"
            ufun_value = negotiator.ufun(session.agreement)
            print(f"{role} - Negotiator '{negotiator.name}' (ID: {negotiator.id}) Utility: {ufun_value:.4f}")
    else:
        print("No agreement reached in the session.")
    #plt = session.plot()
    #plt.show()

def ANAC2015_scenarios_test():
    scenario_names = [
        "group1-university",
        "group2-dinner",
        "group2-new_sporthal",
        "group2-politics",
        "group3-bank_robbery",
        "group4-zoning_plan",
        "group5-car_domain",
        "group6-tram",
        "group7-movie",
        "group8-holiday",
        "group9-killer_robot",
        "group9-vacation",
        "group10-building_construction",
        "group11-car_purchase",
        "group12-symposium"
    ]

    #utilities = {"Micro": [], "AgreeableAgent2018": [], "KakeSoba": []}
    utilities = {"Micro": [], "Atlas3": [], "PonPoko": []}
    #utilities = {"Micro1": [], "Micro2": [], "Micro3": []}

    #proposer_count = {"Micro": 0, "AgreeableAgent2018": 0, "KakeSoba": 0}
    proposer_count = {"Micro": 0, "Atlas3": 0, "PonPoko": 0}
    #proposer_count = {"Micro1": 0, "Micro2": 0, "Micro3": 0}



    total_sessions = 0
    agreements_reached = 0
    for scenario_name in scenario_names:
        print(f"\nRunning scenario: {scenario_name}")
        scenario_path = Path.home() / "negmas" / "scenarios" / "ANAC2015" / scenario_name
        try:
            scenario = Scenario.load(scenario_path)
        except (ET.ParseError, OSError, ValueError) as e:
            print(f"Skipping scenario '{scenario_name}' due to error: {e}")
            continue

        for preference_set in range(len(scenario.ufuns)):
            print(f"  Testing preference set {preference_set + 1} of {len(scenario.ufuns)}")

            session = scenario.make_session(n_steps=100)

            agent1 = MiCRONegotiatorMulti(preferences=scenario.ufuns[preference_set % len(scenario.ufuns)].scale_max(1.0), name="Micro")
            agent2 = Atlas3(preferences=scenario.ufuns[(preference_set + 1) % len(scenario.ufuns)].scale_max(1.0), name="Atlas3")
            agent3 = PonPokoAgent(preferences=scenario.ufuns[(preference_set + 2) % len(scenario.ufuns)].scale_max(1.0), name="PonPoko")
            session.add(agent1)
            session.add(agent2)
            session.add(agent3)

            print(session.run())
            total_sessions = total_sessions + 1
            proposer = session.state.current_proposer

            if session.agreement is not None:
                print("    Final Agreement:", session.agreement)
                agreements_reached = agreements_reached + 1
                for negotiator in session.negotiators:
                    if negotiator.id == proposer:
                        role = "Proposer"
                        proposer_count[negotiator.name] += 1
                    else:
                        role = "Responder"
                    ufun_value = negotiator.ufun(session.agreement)
                    utilities[negotiator.name].append(ufun_value)
                    print(f"    {role} - Negotiator '{negotiator.name}' (ID: {negotiator.id}) Utility: {ufun_value:.4f}")
            else:
                print("    No agreement reached in the session.")
                for negotiator in session.negotiators:
                    utilities[negotiator.name].append(0.0)
            #plt = session.plot()
            #plt.show()

    print("\nUtility Statistics:")
    for name, values in utilities.items():
        if values:
            mean_utility = statistics.mean(values)
            variance = statistics.variance(values) if len(values) > 1 else 0
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            std_error = std_dev / math.sqrt(len(values)) if len(values) > 1 else 0  # Standard Error (SE)
            times_proposer = proposer_count[name]
        else:
            mean_utility, variance, std_dev, std_error, times_proposer = 0, 0, 0, 0, 0

        print(
            f"{name}: Mean = {mean_utility:.4f}, Variance = {variance:.4f}, Std Dev = {std_dev:.4f}, Std Error = {std_error:.4f}, Times Proposer = {times_proposer:.4f}")

    agreement_rate = (agreements_reached / total_sessions) * 100 if total_sessions > 0 else 0
    print(f"\nAgreement reached {agreement_rate:.2f}% of the times.")
    print(f"\nTotal sessions: {total_sessions}")

from datetime import datetime

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    ANAC2015_scenarios_test()
    #run_test()
    end_time = datetime.now()
    print(f"End Time: {end_time}")
    print(f"Execution Time: {end_time - start_time}")

