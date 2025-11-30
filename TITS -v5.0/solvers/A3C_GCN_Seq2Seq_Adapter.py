# solvers/A3C_GCN_Seq2Seq_Adapter.py
"""
Adapter for A3C-GCN-Seq2Seq algorithm to fit into the OLMA framework.
This adapter converts the system state from the VEC_Environment into a format
that the A3C-GCN-Seq2Seq algorithm can process, and converts the algorithm's
output back into the format expected by the environment.
"""
import os
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the base solver class
from solvers.basesolver import BaseSolver

# Try to import the A3C-GCN-Seq2Seq solver components, but handle import errors
try:
    from solvers.a3c_gcn_seq2seq.solver import A3CGcnSeq2SeqSolver
    A3C_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import A3C-GCN-Seq2Seq components: {e}")
    A3C_AVAILABLE = False

class A3C_GCN_Seq2Seq_Adapter(BaseSolver):
    """
    Adapter class that bridges the gap between the OLMA framework and the
    A3C-GCN-Seq2Seq algorithm.
    """
    def __init__(self, env_config: Dict[str, Any], cfg: Dict[str, Any]):
        """
        Initialize the adapter and the underlying A3C-GCN-Seq2Seq solver.
        
        Args:
            env_config: Configuration for the environment
            cfg: Configuration for the solver
        """
        super().__init__(env_config, cfg)
        
        # Store environment configuration
        self.num_vehicles = env_config.get('num_vehicles', 3)
        self.num_servers = env_config.get('num_servers', 2)
        
        # Default resource values
        self.default_power = 0.5  # Default transmission power
        self.default_bandwidth = 1.0  # Default bandwidth allocation
        self.default_frequency = 1e9  # Default CPU frequency
        
        # Initialize the A3C-GCN-Seq2Seq solver only if available
        self.solver = None
        if A3C_AVAILABLE:
            try:
                # Initialize the A3C-GCN-Seq2Seq solver with default parameters
                # Note: You may need to adjust these parameters based on your specific setup
                self.solver_config = {
                    'p_net_setting': {
                        'num_nodes': env_config.get('num_servers', 2),
                        'node_attrs_setting': ['resource', 'extrema'],
                        'link_attrs_setting': ['resource', 'extrema']
                    },
                    'v_sim_setting': {
                        'node_attrs_setting': ['resource'],
                        'link_attrs_setting': ['resource']
                    },
                    'embedding_dim': 64,
                    'lr_actor': 1e-4,
                    'lr_critic': 1e-3
                }
                
                # Merge user-provided configuration
                self.solver_config.update(cfg)
                
                # Create dummy controller, recorder, and counter objects
                # These are required by the A3CGcnSeq2SeqSolver constructor
                self.dummy_controller = type('DummyController', (), {})()
                self.dummy_recorder = type('DummyRecorder', (), {})()
                self.dummy_counter = type('DummyCounter', (), {'count': 0})()
                
                # Initialize the A3C-GCN-Seq2Seq solver
                self.solver = A3CGcnSeq2SeqSolver(
                    controller=self.dummy_controller,
                    recorder=self.dummy_recorder,
                    counter=self.dummy_counter,
                    **self.solver_config
                )
                logger.info("A3C-GCN-Seq2Seq solver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize A3C-GCN-Seq2Seq solver: {e}")
                self.solver = None
        else:
            logger.warning("A3C-GCN-Seq2Seq components not available, will use default assignment")
    
    def solve(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the resource allocation problem using the A3C-GCN-Seq2Seq algorithm.
        
        Args:
            system_state: The current system state from the VEC_Environment
            
        Returns:
            A dictionary containing the resource allocation decisions in the format
            expected by the VEC_Environment.
        """
        try:
            if self.solver is None:
                # Fallback to default assignment if solver is not available
                return self._get_default_decision(system_state)
            
            # Convert the OLMA system state to the format expected by A3C-GCN-Seq2Seq
            instance = self._convert_state_to_instance(system_state)
            
            # Use the A3C-GCN-Seq2Seq solver to find a solution
            solution = self.solver.solve(instance)
            
            # Convert the A3C-GCN-Seq2Seq solution to the format expected by OLMA
            decision = self._convert_solution_to_decision(solution, system_state)
            
            logger.info(f"A3C-GCN-Seq2Seq solver returned assignment: {decision['assignment']}")
            return decision
        except Exception as e:
            logger.error(f"Error in A3C-GCN-Seq2Seq solver: {e}")
            # Fallback to default assignment if any error occurs
            return self._get_default_decision(system_state)
    
    def _convert_state_to_instance(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the OLMA system state to the format expected by A3C-GCN-Seq2Seq.
        
        Args:
            system_state: The current system state from the VEC_Environment
            
        Returns:
            A dictionary containing the instance data in the format expected by A3C-GCN-Seq2Seq.
        """
        # Extract information from the system state
        V_set = system_state.get('V_set', [])
        J_set = system_state.get('J_set', [])
        g = system_state.get('g', np.ones((self.num_vehicles, self.num_servers)))
        tasks = system_state.get('tasks', [{'Din': 0.5, 'Cv': 2e5, 'kv': 0} for _ in V_set])
        
        # Create a dummy physical network (p_net) for the A3C-GCN-Seq2Seq algorithm
        p_net = type('DummyPNet', (), {
            'num_nodes': len(J_set),
            'num_links': len(J_set) * (len(J_set) - 1),  # Fully connected
            'get_node_attrs': lambda self, attr_types: [[1.0] * 2 for _ in J_set],  # Dummy node attributes
            'get_link_attrs': lambda self, attr_types: [[1.0] * 2 for _ in range(len(J_set) * (len(J_set) - 1))]  # Dummy link attributes
        })()
        
        # Create a dummy virtual network (v_net) for the A3C-GCN-Seq2Seq algorithm
        v_net = type('DummyVNet', (), {
            'num_nodes': len(V_set),
            'num_links': len(V_set) * (len(V_set) - 1),  # Fully connected
            'get_node_attrs': lambda self, attr_types: [[task['Din'], task['Cv']] for task in tasks],  # Node attributes from tasks
            'get_link_attrs': lambda self, attr_types: [[1.0] * 2 for _ in range(len(V_set) * (len(V_set) - 1))]  # Dummy link attributes
        })()
        
        # Return the instance in the format expected by A3C-GCN-Seq2Seq
        return {
            'v_net': v_net,
            'p_net': p_net
        }
    
    def _convert_solution_to_decision(self, solution: Any, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the A3C-GCN-Seq2Seq solution to the format expected by OLMA.
        
        Args:
            solution: The solution returned by the A3C-GCN-Seq2Seq solver
            system_state: The current system state from the VEC_Environment
            
        Returns:
            A dictionary containing the resource allocation decisions in the format
            expected by the VEC_Environment.
        """
        V_set = system_state.get('V_set', [])
        J_set = system_state.get('J_set', [])
        
        # Initialize default decisions
        assignment = [0] * len(V_set)
        power = [self.default_power] * len(V_set)
        bandwidth = [[self.default_bandwidth] * len(J_set) for _ in V_set]
        freq = [[self.default_frequency] * len(J_set) for _ in V_set]
        
        # Extract assignment from the solution if available
        try:
            if hasattr(solution, 'node_slots') and solution.node_slots:
                # Map virtual nodes (vehicles) to physical nodes (servers)
                for v_idx, p_idx in enumerate(solution.node_slots):
                    if v_idx < len(assignment) and p_idx < len(J_set):
                        assignment[v_idx] = int(p_idx)
        except Exception as e:
            logger.error(f"Error extracting assignment from solution: {e}")
        
        # Set non-zero values only for the assigned servers
        for v_idx in range(len(V_set)):
            j_idx = assignment[v_idx]
            for j in range(len(J_set)):
                if j != j_idx:
                    bandwidth[v_idx][j] = 0.0
                    freq[v_idx][j] = 0.0
        
        return {
            'assignment': assignment,
            'power': power,
            'bandwidth': bandwidth,
            'freq': freq,
            'debug': {
                'solution_result': solution.result if hasattr(solution, 'result') else False,
                'solution_type': type(solution).__name__
            }
        }
    
    def _get_default_decision(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a default decision when the A3C-GCN-Seq2Seq solver is not available or fails.
        
        Args:
            system_state: The current system state from the VEC_Environment
            
        Returns:
            A dictionary containing default resource allocation decisions
        """
        V_set = system_state.get('V_set', [])
        J_set = system_state.get('J_set', [])
        
        # Simple round-robin assignment
        assignment = [v % len(J_set) for v in range(len(V_set))]
        power = [self.default_power] * len(V_set)
        bandwidth = [[self.default_bandwidth] * len(J_set) for _ in V_set]
        freq = [[self.default_frequency] * len(J_set) for _ in V_set]
        
        # Set non-zero values only for the assigned servers
        for v_idx in range(len(V_set)):
            j_idx = assignment[v_idx]
            for j in range(len(J_set)):
                if j != j_idx:
                    bandwidth[v_idx][j] = 0.0
                    freq[v_idx][j] = 0.0
        
        logger.debug(f"Using default round-robin assignment: {assignment}")
        return {
            'assignment': assignment,
            'power': power,
            'bandwidth': bandwidth,
            'freq': freq,
            'debug': {
                'using_default': True
            }
        }