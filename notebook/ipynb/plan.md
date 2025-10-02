# How can this be useful?

    Predictions --------------------------------------------------------
    1. Take a compound and a topic
    2. Collect all relevant properties
    3. Generate all predictions

    Counterfactuals ----------------------------------------------------
    1. If this property has this value, what would the value of another property be?
    2. If the structure was changed, how would that impact the prediction?
    3. Which tests can I perform to maximally reduce uncertainty while minimizing cost?

    Retrieval ----------------------------------------------------------
    1. Which properties are the most predictive?
    2. What values are the most useful?
    3. <eventually> retrieve all relevant documents

# Collaboration

1. A shared source of truth - biobricks.ai
2. data requests, data hosting - biobricks.ai
3. data scraping - sysrev.com

## What Next?
The property-value transformer currently predicts sequences of property-value pairs. 
The model works, but it is quite restricted. LLMs can operate on arbitrary strings.

The next step is to upgrade every property and value to a string and train an LLM. 
    
    1. Allows model to learn relationships between properties
    2. Allows numeric predictions
    3. Allows integration of full text documents.
