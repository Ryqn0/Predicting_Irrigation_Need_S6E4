from typing import List, Tuple, Optional

import pandas as pd
import great_expectations as gx
from great_expectations.validator.validator import Validator
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.core.batch import Batch


def verify_data(df: pd.DataFrame) -> Tuple[bool, Optional[List[str]]]:
    """
    Using Great Expectations to verify the integrity of the data.
    """

    print("Verifying data integrity using Great Expectations...")
    
    context = gx.get_context(mode="ephemeral")
    validator = Validator(
        execution_engine=PandasExecutionEngine(),
        batches=[
            Batch(data=df, batch_id="data_batch")
        ]
        data_context=context
    )

    # Expectation 1: 'id' column can exist or not but if it exists, it should have only unique values and no nulls
    if 'id' in df.columns:
        validator.expect_column_values_to_not_be_null('id')
        validator.expect_column_values_to_be_unique('id')

    # Expectation 2: 'Soil_Type' column should exist and should have valid values
    validator.expect_column_to_exist('Soil_Type')
    validator.expect_column_values_to_be_in_set('Soil_Type', ['Sandy', 'Clay', 'Silt', 'Loamy'])

    # Expectation 3: 'Crop_Type' column should exist and should have valid values
    validator.expect_column_to_exist('Crop_Type')
    validator.expect_column_values_to_be_in_set('Crop_Type', ['Maize', 'Cotton', 'Wheat', 'Rice', 'Sugarcane', 'Potato'])

    # Expectation 4: 'Crop_Growth_Stage' column should exist and should have valid values
    validator.expect_column_to_exist('Crop_Growth_Stage')
    validator.expect_column_values_to_be_in_set('Crop_Growth_Stage', ['Sowing', 'Harvesting', 'Flowering', 'Vegetative'])

    # Expectation 5: 'Season' column should exist and should have valid values
    validator.expect_column_to_exist('Season')
    validator.expect_column_values_to_be_in_set('Season', ['Rabi', 'Kharif', 'Zaid'])

    # Expectation 6: 'Irrigation_Type' column should exist and should have valid values
    validator.expect_column_to_exist('Irrigation_Type')
    validator.expect_column_values_to_be_in_set('Irrigation_Type', ['Canal', 'Drip', 'Sprinkler', 'Rainfed'])

    # Expectation 7: 'Water_Source' column should exist and should have valid values
    validator.expect_column_to_exist('Water_Source')
    validator.expect_column_values_to_be_in_set('Water_Source', ['River', 'Reservoir', 'Groundwater', 'Rainwater'])

    # Expectation 8: 'Mulching_Used' column should exist and should have valid values
    validator.expect_column_to_exist('Mulching_Used')
    validator.expect_column_values_to_be_in_set('Mulching_Used', ['Yes', 'No'])

    # Expectation 9: 'Region' column should exist and should have valid values
    validator.expect_column_to_exist('Region')
    validator.expect_column_values_to_be_in_set('Region', ['North', 'South', 'East', 'West', 'Central'])

    # Expectation 10: 'Irrigation_Need' column should exist and should have valid values (training data only)
    if 'Irrigation_Need' in df.columns:
        validator.expect_column_to_exist('Irrigation_Need')
        validator.expect_column_values_to_be_in_set('Irrigation_Need', ['Low', 'Medium', 'High'])

    # Expectation 11: "Soil_pH" column should exist and should have values between 3.5 and 9.0
    validator.expect_column_to_exist('Soil_pH')
    validator.expect_column_values_to_be_between('Soil_pH', min_value=3.5, max_value=9.0)

    # Expectation 12: "Soil_Moisture" column should exist and should have non-negative values
    validator.expect_column_to_exist('Soil_Moisture')
    validator.expect_column_values_to_be_between('Soil_Moisture', min_value=0, max_value=100)

    # Expectation 13: "Organic_Carbon" column should exist and should have values between 0 to 2
    validator.expect_column_to_exist('Organic_Carbon')
    validator.expect_column_values_to_be_between('Organic_Carbon', min_value=0, max_value=2)

    # Expectation 14: "Electrical_Conductivity" column should exist and should have values between 0 and 4 (according to train data)
    validator.expect_column_to_exist('Electrical_Conductivity')
    validator.expect_column_values_to_be_between('Electrical_Conductivity', min_value=0, max_value=4)

    # Expectation 15: "Temperature_C" column should exist and should have values between 0 and 50 (according to train data)
    validator.expect_column_to_exist('Temperature_C')
    validator.expect_column_values_to_be_between('Temperature_C', min_value=0, max_value=50)

    # Expectation 16: "Humidity" column should exist and should have values between 0 and 100
    validator.expect_column_to_exist('Humidity')
    validator.expect_column_values_to_be_between('Humidity', min_value=0, max_value=100)

    # Expectation 17: "Rainfall_mm" column should exist and should have values between 0 and 2600 (according to train data)
    validator.expect_column_to_exist('Rainfall_mm')
    validator.expect_column_values_to_be_between('Rainfall_mm', min_value=0, max_value=2600)

    # Expectation 18: "Sunlight_Hours" column should exist and should have values between 3 and 16 (according to train data)
    validator.expect_column_to_exist('Sunlight_Hours')
    validator.expect_column_values_to_be_between('Sunlight_Hours', min_value=3, max_value=16)

    # Expectation 19: "Wind_Speed_kmh" column should exist and should have values between 0 and 25 (according to train data)
    validator.expect_column_to_exist('Wind_Speed_kmh')
    validator.expect_column_values_to_be_between('Wind_Speed_kmh', min_value=0, max_value=25)

    # Expectation 20: "Field_Area_hectare" column should exist and should have valid values between 0 and 20 (according to train data)
    validator.expect_column_to_exist('Field_Area_hectare')
    validator.expect_column_values_to_be_between('Field_Area_hectare', min_value=0, max_value=20)

    # Expectation 21: "Previous_Irrigation_mm" column should exist and should have valid values between 0 and 130 (according to train data)
    validator.expect_column_to_exist('Previous_Irrigation_mm')
    validator.expect_column_values_to_be_between('Previous_Irrigation_mm', min_value=0, max_value=130)

    # Check if all expectations passed
    results = validator.validate()
    if not results["success"]:
        print("Data verification failed with the following issues:")
        for result in results["results"]:
            if not result["success"]:
                print(f"- Expectation: {result['expectation_config']['expectation_type']}")
                print(f"  Column: {result['expectation_config']['kwargs'].get('column')}")
                print(f"  Details: {result['result']}")
        return False, [f"{result['expectation_config']['expectation_type']} on column {result['expectation_config']['kwargs'].get('column')}" for result in results["results"] if not result["success"]]

    return True