"""
Call Center Performance Analysis

"""

import simpy
import numpy as np
import pandas as pd
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

print("\n")
print("CALL CENTER PERFORMANCE ANALYSIS")
print()

# System parameters for different periods
PARAMETERS = {
    'morning': {
        'name': 'Morning (8 AM - 12 PM)',
        'duration': 240,
        'arrival_rate': 20,
        'service_time_mean': 5,
        'num_agents': 3,
        'description': 'Low demand period - mostly billing inquiries'
    },
    'afternoon': {
        'name': 'Afternoon (12 PM - 6 PM)', 
        'duration': 360,
        'arrival_rate': 35,
        'service_time_mean': 5.5,
        'num_agents': 5,
        'description': 'Moderate demand - mixed call types'
    },
    'evening': {
        'name': 'Evening (6 PM - 10 PM)',
        'duration': 240,
        'arrival_rate': 60,
        'service_time_mean': 6,
        'num_agents': 6,
        'description': 'Peak demand - technical + billing + account issues'
    }
}

PATIENCE_MEAN = 10  # how long customers wait before hanging up (minutes)

print("***********************System Parameters***********************")
print()
for period, params in PARAMETERS.items():
    print(f"\n{params['name']}:")
    print(f"  Arrival Rate: {params['arrival_rate']} calls/hour")
    print(f"  Avg Service Time: {params['service_time_mean']} minutes")
    print(f"  Agents on Duty: {params['num_agents']}")
    print(f"  {params['description']}")

print(f"\nCustomer Patience: {PATIENCE_MEAN} min average")
print()

# Queueing theory formulas for multi-server system
def calculate_queue_metrics(arrival_rate_per_hour, service_time_min, num_agents):
    """Calculate theoretical wait times using M/M/c queue formulas"""
    
    lam = arrival_rate_per_hour / 60.0
    mu = 1.0 / service_time_min
    
    # Traffic intensity
    rho = lam / (num_agents * mu)
    traffic_intensity = lam / mu
    
    # Check if system is stable
    if rho >= 1:
        return {
            'stable': False,
            'rho': rho,
            'message': 'OVERLOADED - Queue grows infinitely!'
        }
    
    # Calculate waiting probability using queueing theory
    c = num_agents
    A = traffic_intensity
    
    probability_sum = sum([(A**k) / factorial(k) for k in range(c)])
    blocking_probability = (A**c / factorial(c)) / probability_sum
    
    prob_wait = blocking_probability / (1 - rho + blocking_probability * rho)
    
    avg_wait = prob_wait * (service_time_min / (c * (1 - rho)))
    avg_wait_all = prob_wait * avg_wait
    
    return {
        'stable': True,
        'rho': rho,
        'utilization_pct': rho * 100,
        'traffic_intensity': traffic_intensity,
        'prob_wait': prob_wait,
        'avg_wait_all': avg_wait_all,
        'avg_wait_if_wait': avg_wait
    }


print("\n***********Theoretical Performance (Mathematical Model)***********\n")
print()

analytical_results = {}
for period, params in PARAMETERS.items():
    result = calculate_queue_metrics(params['arrival_rate'], params['service_time_mean'], 
                      params['num_agents'])
    analytical_results[period] = result
    
    print(f"\n{params['name']}:")
    if not result['stable']:
        print(f"  {result['message']}")
        print(f"  Traffic Intensity (ρ): {result['rho']:.2f}")
    else:
        print(f"  Traffic Intensity (ρ): {result['rho']:.3f}")
        print(f"  Agent Utilization: {result['utilization_pct']:.1f}%")
        print(f"  Probability Call Waits: {result['prob_wait']*100:.1f}%")
        print(f"  Average Wait Time: {result['avg_wait_all']:.2f} minutes")
        
        if result['avg_wait_all'] > 3:
            print(f"PROBLEM: Wait time exceeds 3-minute target!")

# Simulation classes
class CallCenter:
    """Represents the call center with agents"""
    
    def __init__(self, env, num_agents, service_time_mean):
        self.env = env
        self.agents = simpy.Resource(env, num_agents)
        self.service_time_mean = service_time_mean
        
        # Track metrics
        self.calls_arrived = 0
        self.calls_served = 0
        self.calls_abandoned = 0
        self.wait_times = []
        self.service_times = []
        self.queue_lengths = []
        
    def serve_call(self, call_id, arrival_time, patience):
        """Handle a single customer call"""
        self.calls_arrived += 1
        
        queue_len = len(self.agents.queue)
        self.queue_lengths.append(queue_len)
        
        with self.agents.request() as request:
            # Wait for agent or give up if too long
            result = yield request | self.env.timeout(patience)
            
            wait_time = self.env.now - arrival_time
            
            if request in result:
                # Got an agent
                self.wait_times.append(wait_time)
                
                service_time = np.random.exponential(self.service_time_mean)
                self.service_times.append(service_time)
                yield self.env.timeout(service_time)
                
                self.calls_served += 1
            else:
                # Customer hung up
                self.calls_abandoned += 1

def generate_calls(env, call_center, arrival_rate_per_hour, duration, patience_mean):
    """Generate random call arrivals"""
    call_id = 0
    end_time = env.now + duration
    
    arrival_rate_per_min = arrival_rate_per_hour / 60.0
    
    while env.now < end_time:
        inter_arrival = np.random.exponential(1.0 / arrival_rate_per_min)
        yield env.timeout(inter_arrival)
        
        if env.now < end_time:
            patience = np.random.exponential(patience_mean)
            call_id += 1
            env.process(call_center.serve_call(call_id, env.now, patience))

def run_simulation(period_name, params, patience_mean):
    """Run the simulation for one period"""
    env = simpy.Environment()
    
    call_center = CallCenter(env, params['num_agents'], 
                             params['service_time_mean'])
    
    env.process(generate_calls(env, call_center, params['arrival_rate'],
                               params['duration'], patience_mean))
    
    env.run()
    
    # Calculate results
    if len(call_center.wait_times) > 0:
        avg_wait = np.mean(call_center.wait_times)
        max_wait = np.max(call_center.wait_times)
    else:
        avg_wait = 0
        max_wait = 0
    
    total_calls = call_center.calls_arrived
    abandonment_rate = (call_center.calls_abandoned / total_calls * 100) if total_calls > 0 else 0
    
    total_service_time = sum(call_center.service_times)
    total_agent_time = params['duration'] * params['num_agents']
    utilization = (total_service_time / total_agent_time * 100) if total_agent_time > 0 else 0
    
    return {
        'calls_arrived': call_center.calls_arrived,
        'calls_served': call_center.calls_served,
        'calls_abandoned': call_center.calls_abandoned,
        'abandonment_rate': abandonment_rate,
        'avg_wait': avg_wait,
        'max_wait': max_wait,
        'utilization': utilization,
        'wait_times': call_center.wait_times,
        'queue_lengths': call_center.queue_lengths
    }


print("\n********************Simulation Results********************")
print("\nRunning simulations...")
print("(captures abandonment and random variations)")
print()

simulation_results = {}

for period, params in PARAMETERS.items():
    result = run_simulation(period, params, PATIENCE_MEAN)
    simulation_results[period] = result
    
    print(f"\n{params['name']}:")
    print(f"  Calls Arrived: {result['calls_arrived']}")
    print(f"  Calls Served: {result['calls_served']}")
    print(f"  Calls Abandoned: {result['calls_abandoned']} ({result['abandonment_rate']:.1f}%)")
    print(f"  Avg Wait Time: {result['avg_wait']:.2f} minutes")
    print(f"  Max Wait Time: {result['max_wait']:.2f} minutes")
    print(f"  Agent Utilization: {result['utilization']:.1f}%")
    
    if result['avg_wait'] > 3:
        print(f"  Wait time exceeds target!")
    if result['abandonment_rate'] > 5:
        print(f"  High abandonment rate!")

# Test different Agents numbers
print("\n*********************Testing Solutions (Sensitivity Analysis)*********************")
print("\nTrying different evening agent staffing levels:")
print()

evening_scenarios = []
agent_counts = [6, 7, 8, 9, 10]

for num_agents in agent_counts:
    theoretical = calculate_queue_metrics(PARAMETERS['evening']['arrival_rate'],
                         PARAMETERS['evening']['service_time_mean'],
                         num_agents)
    
    test_params = PARAMETERS['evening'].copy()
    test_params['num_agents'] = num_agents
    sim_result = run_simulation('evening_test', test_params, PATIENCE_MEAN)
    
    evening_scenarios.append({
        'agents': num_agents,
        'theoretical_wait': theoretical['avg_wait_all'] if theoretical['stable'] else 999,
        'simulated_wait': sim_result['avg_wait'],
        'abandonment': sim_result['abandonment_rate'],
        'utilization': sim_result['utilization'],
        'calls_served': sim_result['calls_served']
    })
    
    print(f"\n{num_agents} Agents:")
    print(f"  Wait Time: {sim_result['avg_wait']:.2f} min")
    print(f"  Abandonment: {sim_result['abandonment_rate']:.1f}%")
    print(f"  Utilization: {sim_result['utilization']:.1f}%")
    
    meets_wait = sim_result['avg_wait'] < 3
    meets_abandon = sim_result['abandonment_rate'] < 5
    meets_util = 70 <= sim_result['utilization'] <= 85
    
    if meets_wait and meets_abandon and meets_util:
        print(f"OPTIMAL: Meets all targets!")

print("\n")
print("SIMULATION COMPLETE")
print("\n")