from src.negmas.gb.negotiators.micro_multi import MiCRONegotiatorMulti
from pathlib import Path
from negmas.inout import Scenario
from negmas.genius.gnegotiators import Atlas3, PonPokoAgent, CaduceusDC16, AgreeableAgent2018
import statistics
import math
import xml.etree.ElementTree as ET
from itertools import combinations_with_replacement
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import random

def create_agents():
    return [
        AgreeableAgent2018(name="Agreeable"),
        PonPokoAgent(name="PonPoko"),
        Atlas3(name="Atlas3"),
        MiCRONegotiatorMulti(name="Micro"),
    ]

def ANAC2015_scenarios_test():
    scenario_names = [
        "group1-university",
        "group2-dinner",
        "group2-politics",
        "group3-bank_robbery",
        "group5-car_domain",
        "group6-tram",
        "group8-holiday",
        "group9-killer_robot",
        "group9-vacation",
        "group11-car_purchase"]


    utilities = {"Agreeable": [], "PonPoko": [], "Atlas3": [], "Micro": []}
    utilities_on_agreement = {"Agreeable": [], "PonPoko": [], "Atlas3": [], "Micro": []}
    total_sessions = {"Agreeable": 0, "PonPoko": 0, "Atlas3": 0, "Micro": 0}
    agreements_reached = {"Agreeable": 0, "PonPoko": 0, "Atlas3": 0, "Micro": 0}
    performance_against_pairs = {}

    for scenario_name in scenario_names:
        print(f"\nRunning scenario: {scenario_name}")
        scenario_path = Path.home() / "negmas" / "scenarios" / "ANAC2015" / scenario_name
        try:
            scenario = Scenario.load(scenario_path)
        except (ET.ParseError, OSError, ValueError) as e:
            print(f"Skipping scenario '{scenario_name}' due to error: {e}")
            continue
        agents = create_agents()
        agent_combinations = list(combinations_with_replacement(agents, 3))

        for index, (agent1_class, agent2_class, agent3_class) in enumerate(agent_combinations, start=1):
            print(
                f"Iteration {index}/{len(agent_combinations)}: Running session for {agent1_class.name}, {agent2_class.name}, {agent3_class.name}")

            agent1 = type(agent1_class)(name=agent1_class.name)
            agent2 = type(agent2_class)(name=agent2_class.name)
            agent3 = type(agent3_class)(name=agent3_class.name)

            session = scenario.make_session(time_limit=35)

            session.add(agent1, preferences=scenario.ufuns[0].scale_max(1.0))
            session.add(agent2, preferences=scenario.ufuns[1].scale_max(1.0))
            session.add(agent3, preferences=scenario.ufuns[2].scale_max(1.0))

            print(session.run())
            process_session_results(session, utilities, utilities_on_agreement,
                                    total_sessions, agreements_reached, performance_against_pairs)

    print_utility_statistics(utilities, utilities_on_agreement, total_sessions, agreements_reached)
    print_performance_against_pairs(performance_against_pairs)
    return performance_against_pairs


def process_session_results(session, utilities, utilities_on_agreement,
                            total_sessions, agreements_reached, performance_against_pairs):
    for negotiator in session.negotiators:
        agent_name = negotiator.name
        utility = negotiator.ufun(session.agreement) if session.agreement else negotiator.reserved_value

        other_agents = [a.name for a in session.negotiators if a.id != negotiator.id]
        pair = tuple(sorted(other_agents))
        key = (pair, agent_name)

        performance_against_pairs.setdefault(key, []).append(utility)
        total_sessions[agent_name] += 1

    proposer = session.state.current_proposer

    if session.agreement is not None:
        print("    Final Agreement:", session.agreement)
        for negotiator in session.negotiators:
            role = "Proposer" if negotiator.id == proposer else "Responder"
            ufun_value = negotiator.ufun(session.agreement)
            utilities[negotiator.name].append(ufun_value)
            utilities_on_agreement[negotiator.name].append(ufun_value)
            agreements_reached[negotiator.name] += 1
            print(f"    {role} - Negotiator '{negotiator.name}' (ID: {negotiator.id}) Utility: {ufun_value:.4f}")
    else:
        print("    No agreement reached in the session.")
        for negotiator in session.negotiators:
            utilities[negotiator.name].append(negotiator.reserved_value)
            print("NO AGREEMENT")
            print(negotiator.reserved_value)

def print_utility_statistics(utilities, utilities_on_agreement, total_sessions, agreements_reached):
    print("\n### Utility Statistics ###")
    for name, values in utilities.items():
        if values:
            mean_utility = statistics.mean(values)
            utility_on_agreement = statistics.mean(utilities_on_agreement[name]) if name in utilities_on_agreement else 0
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            std_error = std_dev / math.sqrt(len(values)) if len(values) > 1 else 0
            agreement = (agreements_reached[name] / total_sessions[name]) * 100 if total_sessions[name] > 0 else 0
        else:
            mean_utility, utility_on_agreement, std_error, agreement = 0, 0, 0, 0

        print(f"{name}: Mean Utility = {mean_utility:.4f}, Utility_on_agreement = {utility_on_agreement:.4f},"
              f" Std Error = {std_error:.4f}, Agreement = {agreement:.4f}%")

def print_performance_against_pairs(performance_against_pairs):
    print("\n### Sorted Performance Against Pairs ###")

    pair_to_agents = {}
    for (pair, agent), values in performance_against_pairs.items():
        mean_utility = sum(values) / len(values)
        pair_to_agents.setdefault(pair, []).append((agent, mean_utility))

    for pair in sorted(pair_to_agents.keys()):
        print(f"\nAgainst pair {pair}:")
        sorted_agents = sorted(pair_to_agents[pair], key=lambda x: x[1], reverse=True)
        for agent, mean_utility in sorted_agents:
            print(f"  Agent {agent} → average utility: {mean_utility:.4f}")

def get_best_agents_against_pairs(performance_against_pairs):
    best_agents = {}

    pair_to_agents = {}
    for (pair, agent), values in performance_against_pairs.items():
        mean_utility = sum(values) / len(values)
        pair_to_agents.setdefault(pair, []).append((agent, mean_utility))

    for pair, agent_utils in pair_to_agents.items():
        agent_utils.sort(key=lambda x: x[1], reverse=True)
        best_agents[pair] = agent_utils[0][0]

    return best_agents

def simulate_best_response_dynamics(agent_names, best_agents_by_pair, max_steps=20, seed=None):
    if seed is not None:
        random.seed(seed)

    state = tuple(random.choices(agent_names, k=3))
    seen_states = set()
    transitions = []

    for step in range(max_steps):
        seen_states.add(state)
        current_state = list(state)

        for player in range(3):
            others = tuple(sorted([current_state[i] for i in range(3) if i != player]))
            best_response = best_agents_by_pair.get(others)

            if best_response and best_response != current_state[player]:
                next_state = current_state.copy()
                next_state[player] = best_response
                next_state = tuple(next_state)
                transitions.append((state, next_state, player))
                state = next_state
                break
        else:
            break

    return transitions, state

def plot_nash_equilibrium_graph(transitions, final_state):
    G = nx.DiGraph()

    for from_state, to_state, player in transitions:
        G.add_edge(from_state, to_state, label=f"P{player + 1}")

    #pos = nx.spring_layout(G, seed=42, k=2.0, iterations=300, scale=5.0)
    pos = nx.kamada_kawai_layout(G)
    node_colors = ['lightgreen' if node == final_state else 'lightblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, edgecolors='black')

    nx.draw_networkx_edges(
        G,
        pos,
        connectionstyle="arc3,rad=0.1",
        arrows=True,
        arrowstyle='-|>',
        arrowsize=10,
        width=1.5,
        edge_color='gray',
        min_source_margin=15,
        min_target_margin=15
    )
    nx.draw_networkx_labels(G, pos, font_size=8, verticalalignment='center')

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5)

    plt.title("Best-Response Dynamics → Nash Equilibrium", fontsize=14)
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.margins(x=0.3, y=0.3)

    plt.show()

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    performance_against_pairs = ANAC2015_scenarios_test()
    #performance_against_pairs = #performance pairs txt
    agent_names = ["Agreeable", "PonPoko", "Atlas3", "Micro"]
    best_agents_by_pair = get_best_agents_against_pairs(performance_against_pairs)

    transitions, final_state = simulate_best_response_dynamics(agent_names, best_agents_by_pair)
    plot_nash_equilibrium_graph(transitions, final_state)

    end_time = datetime.now()
    print(f"End Time: {end_time}")
    print(f"Execution Time: {end_time - start_time}")
