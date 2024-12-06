from web3 import Web3

# Set up connection to local Ethereum node 
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

# Ensure that the Web3 connection is successful
if not w3.is_connected():  
    print("Failed to connect to Ethereum node")
    exit()

contract_source_code = '''
pragma solidity ^0.8.0;

contract EnergyManagement {
    struct EnergyRecord {
        uint timestamp;
        uint energyUsed;
        uint renewablePercentage;
    }

    EnergyRecord[] public records;

    function addRecord(uint _energyUsed, uint _renewablePercentage) public {
        records.push(EnergyRecord(block.timestamp, _energyUsed, _renewablePercentage));
    }

    function getRecord(uint index) public view returns (uint, uint, uint) {
        EnergyRecord memory record = records[index];
        return (record.timestamp, record.energyUsed, record.renewablePercentage);
    }
}
'''

# Contract ABI (replace with your actual ABI after deployment)
contract_abi = [
    {
        "constant": False,
        "inputs": [
            {"name": "_energyUsed", "type": "uint256"},
            {"name": "_renewablePercentage", "type": "uint256"}
        ],
        "name": "addRecord",
        "outputs": [],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [{"name": "index", "type": "uint256"}],
        "name": "getRecord",
        "outputs": [
            {"name": "", "type": "uint256"},
            {"name": "", "type": "uint256"},
            {"name": "", "type": "uint256"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
]

# Replace with the actual deployed contract address
contract_address = '0xYourContractAddress'  # Replace with your actual contract address

# Instantiate contract with the ABI and address
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Check if the contract is valid
if contract is None:
    print("Failed to instantiate contract. Please check the address and ABI.")
    exit()

# Function to log energy usage to the Ethereum contract
def log_energy_usage(energy_used, renewable_percentage):
    try:
        # Get the default account (replace with the correct account if needed)
        sender_account = w3.eth.accounts[0]

        # Prepare and send the transaction
        tx_hash = contract.functions.addRecord(energy_used, renewable_percentage).transact({'from': sender_account})

        # Wait for the transaction receipt to confirm the transaction
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        # Check the transaction status and log the energy usage
        if receipt['status'] == 1:
            print(f"Logged energy usage: {energy_used}, Renewable: {renewable_percentage}%")
        else:
            print("Transaction failed. Please check the contract or transaction details.")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage: Log energy usage (replace with actual data)
log_energy_usage(energy_used=100, renewable_percentage=70)
