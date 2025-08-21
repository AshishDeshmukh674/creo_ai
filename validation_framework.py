# Comprehensive Creo Command Validation Framework
# Ensures generated trail commands are syntactically correct and executable

import re
import json
from typing import List, Dict, Tuple, Optional
from enum import Enum

class ValidationResult(Enum):
    VALID = "valid"
    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_ERROR = "semantic_error"
    EXECUTION_ERROR = "execution_error"

class CreoCommandValidator:
    def __init__(self):
        # Load Creo command specifications
        self.command_specs = self.load_command_specifications()
        self.valid_commands = set(self.command_specs.keys())
        self.parameter_patterns = self.load_parameter_patterns()
        
    def load_command_specifications(self) -> Dict:
        """Load valid Creo commands and their parameter requirements"""
        return {
            "ProCmdDashboardActivate": {
                "parameters": [],
                "required": True,
                "context": ["startup", "modeling"]
            },
            "ProCmdModelNew": {
                "parameters": [],
                "required": False,
                "context": ["modeling"]
            },
            "ProCmdExtrudeDashboard": {
                "parameters": ["depth"],
                "required": True,
                "context": ["sketch_active"],
                "parameter_types": {"depth": "numeric"}
            },
            "ProCmdDashboardAccept": {
                "parameters": [],
                "required": True,
                "context": ["operation_pending"]
            }
        }
    
    def validate_trail_file(self, trail_content: str) -> Tuple[ValidationResult, List[str]]:
        """
        Comprehensive validation of entire trail file
        Returns: (ValidationResult, list_of_errors)
        """
        errors = []
        
        # 1. SYNTAX VALIDATION
        syntax_errors = self.validate_syntax(trail_content)
        errors.extend(syntax_errors)
        
        # 2. SEMANTIC VALIDATION
        semantic_errors = self.validate_semantics(trail_content)
        errors.extend(semantic_errors)
        
        # 3. EXECUTION FLOW VALIDATION
        flow_errors = self.validate_execution_flow(trail_content)
        errors.extend(flow_errors)
        
        # 4. PARAMETER VALIDATION
        param_errors = self.validate_parameters(trail_content)
        errors.extend(param_errors)
        
        if not errors:
            return ValidationResult.VALID, []
        elif any("syntax" in error.lower() for error in errors):
            return ValidationResult.SYNTAX_ERROR, errors
        elif any("semantic" in error.lower() for error in errors):
            return ValidationResult.SEMANTIC_ERROR, errors
        else:
            return ValidationResult.EXECUTION_ERROR, errors
    
    def validate_syntax(self, content: str) -> List[str]:
        """Validate command syntax and format"""
        errors = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check command format: ~ Command `CommandName`
            command_pattern = r'^~ Command `([^`]+)`$'
            activate_pattern = r'^~ Activate `([^`]+)`$'
            input_pattern = r'^~ Input `([^`]+)`( `([^`]+)`)*$'
            comment_pattern = r'^! (.+)$'
            
            if not (re.match(command_pattern, line) or 
                   re.match(activate_pattern, line) or 
                   re.match(input_pattern, line) or 
                   re.match(comment_pattern, line)):
                errors.append(f"Line {i}: Invalid syntax format: {line}")
        
        return errors
    
    def validate_semantics(self, content: str) -> List[str]:
        """Validate command semantics and relationships"""
        errors = []
        commands = self.extract_commands(content)
        
        # Check if commands exist in Creo
        for line_num, command in commands:
            if command not in self.valid_commands:
                errors.append(f"Line {line_num}: Unknown command '{command}'")
        
        return errors
    
    def validate_execution_flow(self, content: str) -> List[str]:
        """Validate logical flow of commands"""
        errors = []
        commands = self.extract_commands(content)
        context_stack = ["startup"]  # Track current context
        
        for line_num, command in commands:
            spec = self.command_specs.get(command, {})
            required_context = spec.get("context", [])
            
            # Check if command can be executed in current context
            if required_context and not any(ctx in context_stack for ctx in required_context):
                errors.append(
                    f"Line {line_num}: Command '{command}' cannot be executed "
                    f"in context {context_stack}. Required: {required_context}"
                )
            
            # Update context based on command
            self.update_context(command, context_stack)
        
        return errors
    
    def validate_parameters(self, content: str) -> List[str]:
        """Validate command parameters and values"""
        errors = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Extract Input commands with parameters
            input_match = re.match(r'^~ Input `([^`]+)`(.*)', line)
            if input_match:
                param_value = input_match.group(1)
                additional_params = input_match.group(2)
                
                # Validate numeric parameters
                if self.should_be_numeric(param_value):
                    try:
                        float(param_value)
                    except ValueError:
                        errors.append(f"Line {i}: Expected numeric value, got '{param_value}'")
                
                # Validate parameter ranges
                range_errors = self.validate_parameter_ranges(param_value)
                errors.extend([f"Line {i}: {error}" for error in range_errors])
        
        return errors
    
    def extract_commands(self, content: str) -> List[Tuple[int, str]]:
        """Extract command names with line numbers"""
        commands = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            command_match = re.match(r'^~ Command `([^`]+)`$', line)
            if command_match:
                commands.append((i, command_match.group(1)))
        
        return commands

# AUTOMATED TESTING FRAMEWORK
class CreoCommandTestSuite:
    def __init__(self):
        self.validator = CreoCommandValidator()
        self.test_cases = self.load_test_cases()
    
    def load_test_cases(self) -> Dict:
        """Load comprehensive test cases"""
        return {
            "valid_cube_creation": {
                "input": "Create a 50mm cube",
                "expected_commands": ["ProCmdDashboardActivate", "ProCmdModelNew"],
                "should_pass": True
            },
            "invalid_syntax": {
                "input": "Create invalid command",
                "expected_output": "Command InvalidCommand",  # Wrong format
                "should_pass": False,
                "expected_error": "syntax_error"
            },
            "missing_context": {
                "input": "Extrude without sketch",
                "expected_commands": ["ProCmdExtrudeDashboard"],  # No sketch context
                "should_pass": False,
                "expected_error": "execution_error"
            },
            "invalid_parameters": {
                "input": "Create cube with size ABC",
                "expected_parameters": ["ABC"],  # Non-numeric
                "should_pass": False,
                "expected_error": "parameter_error"
            }
        }
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all test cases and return results"""
        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "details": {}
        }
        
        for test_name, test_case in self.test_cases.items():
            try:
                result = self.run_single_test(test_name, test_case)
                results["details"][test_name] = result
                
                if result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["details"][test_name] = {
                    "passed": False,
                    "error": f"Test execution failed: {str(e)}"
                }
                results["failed"] += 1
        
        return results
    
    def run_single_test(self, test_name: str, test_case: Dict) -> Dict:
        """Run individual test case"""
        # Generate Creo commands from input
        generated_commands = self.generate_commands(test_case["input"])
        
        # Validate generated commands
        validation_result, errors = self.validator.validate_trail_file(generated_commands)
        
        # Check if result matches expectation
        should_pass = test_case.get("should_pass", True)
        actually_passed = validation_result == ValidationResult.VALID
        
        test_passed = should_pass == actually_passed
        
        return {
            "passed": test_passed,
            "generated_commands": generated_commands,
            "validation_result": validation_result.value,
            "errors": errors,
            "expected_to_pass": should_pass,
            "actually_passed": actually_passed
        }

# INTEGRATION WITH YOUR MODEL
class ModelWithValidation:
    def __init__(self, model_path: str):
        self.model = NL2Trail(model_path)  # Your existing model
        self.validator = CreoCommandValidator()
        self.test_suite = CreoCommandTestSuite()
    
    def generate_and_validate(self, nl_input: str) -> Dict:
        """Generate commands and validate them"""
        # Generate using your model
        generated_commands = self.model.generate(nl_input)
        
        # Validate output
        validation_result, errors = self.validator.validate_trail_file(generated_commands)
        
        # If validation fails, attempt correction
        if validation_result != ValidationResult.VALID:
            corrected_commands = self.attempt_correction(generated_commands, errors)
            if corrected_commands:
                validation_result, errors = self.validator.validate_trail_file(corrected_commands)
                generated_commands = corrected_commands
        
        return {
            "input": nl_input,
            "output": generated_commands,
            "validation_result": validation_result.value,
            "errors": errors,
            "is_valid": validation_result == ValidationResult.VALID
        }
    
    def attempt_correction(self, commands: str, errors: List[str]) -> Optional[str]:
        """Attempt to automatically correct common errors"""
        # Implement common corrections
        corrected = commands
        
        # Fix syntax errors
        for error in errors:
            if "Invalid syntax format" in error:
                # Apply syntax corrections
                corrected = self.fix_syntax_errors(corrected)
        
        return corrected if corrected != commands else None
