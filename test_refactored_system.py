
import asyncio
import os
import sys
import json

# Add current directory to path
sys.path.append(os.getcwd())

from car_dealer_agents import run_orchestrator, init_booking, _FORM_STATES

async def test_system():
    print("=== TESTING REFACTORED SYSTEM ===")
    
    # Test 1: Greeting (Rule-based)
    print("\n[Test 1] Greeting")
    res, logs = await run_orchestrator("Hello", dealer_name="Lookers")
    print(f"Response: {res}")
    assert "Lookers assistant" in res
    print("✅ Greeting passed")

    # Test 2: Intent Classification & Routing (Knowledge)
    print("\n[Test 2] Knowledge Query")
    # We need to mock the KB or ensure it handles "no info" gracefully if no KB loaded
    # But run_orchestrator loads KB.
    # Let's assume "Lookers" has some data or we get "not available"
    res, logs = await run_orchestrator("What are your opening hours?", dealer_name="Lookers")
    print(f"Response: {res}")
    # It might use date/time tool or knowledge agent. "opening hours" is usually knowledge or date/time.
    # If intent is date_time_query, it uses tool.
    # Let's try a pure knowledge query.
    res, logs = await run_orchestrator("What warranty do you offer?", dealer_name="Lookers")
    print(f"Response: {res}")
    # Should be from Knowledge Agent
    print("✅ Knowledge passed (response received)")

    # Test 3: Booking Flow
    print("\n[Test 3] Booking Flow")
    
    # 3a. Start Booking
    print("User: I want to book a service")
    res, logs = await run_orchestrator("I want to book a service", dealer_name="Lookers")
    print(f"Agent: {res}")
    print("Logs:", logs)
    assert "date" in res.lower()
    
    # 3b. Provide Date
    print("User: Next Monday")
    res, logs = await run_orchestrator("Next Monday", dealer_name="Lookers")
    print(f"Agent: {res}")
    assert "time" in res.lower()
    
    # 3c. Provide Time
    print("User: 10:00")
    res, logs = await run_orchestrator("10:00", dealer_name="Lookers")
    print(f"Agent: {res}")
    assert "name" in res.lower()
    
    # 3d. Provide Name
    print("User: John Doe")
    res, logs = await run_orchestrator("John Doe", dealer_name="Lookers")
    print(f"Agent: {res}")
    assert "phone" in res.lower()
    
    # 3e. Provide Phone
    print("User: 07123456789")
    res, logs = await run_orchestrator("07123456789", dealer_name="Lookers")
    print(f"Agent: {res}")
    assert "make" in res.lower()
    
    # 3f. Provide Vehicle Make
    print("User: BMW")
    res, logs = await run_orchestrator("BMW", dealer_name="Lookers")
    print(f"Agent: {res}")
    assert "model" in res.lower()

    # 3g. Provide Vehicle Model
    print("User: X5")
    res, logs = await run_orchestrator("X5", dealer_name="Lookers")
    print(f"Agent: {res}")
    # Optional fields might be skipped or asked. 
    # Current logic: "Check optional fields (simplified: just ask all for now...)"
    # So it will ask for Email.
    assert "email" in res.lower() or "confirm" in res.lower()

    # 3h. Provide Email (if asked)
    if "email" in res.lower():
        print("User: john@example.com")
        res, logs = await run_orchestrator("john@example.com", dealer_name="Lookers")
        print(f"Agent: {res}")
    
    # Should be confirmation now (or more optionals)
    # Let's just loop until confirmation
    while "confirm" not in res.lower() and "finalized" not in res.lower():
        print("User: No notes") # Skip optional
        res, logs = await run_orchestrator("No notes", dealer_name="Lookers")
        print(f"Agent: {res}")

    if "confirm" in res.lower():
        print("User: Yes")
        res, logs = await run_orchestrator("Yes", dealer_name="Lookers")
        print(f"Agent: {res}")
        assert "Great" in res or "booking" in res
        print("✅ Booking passed")
    else:
        print("⚠️ Booking flow incomplete or different steps")

    # Test 4: Dealer Isolation
    print("\n[Test 4] Dealer Isolation")
    res, logs = await run_orchestrator("What does Sytner offer?", dealer_name="Lookers")
    print(f"Response: {res}")
    assert "don't have information about Sytner" in res
    print("✅ Isolation passed")

if __name__ == "__main__":
    from car_dealer_agents import async_run_orchestrator as run_orchestrator
    asyncio.run(test_system())
