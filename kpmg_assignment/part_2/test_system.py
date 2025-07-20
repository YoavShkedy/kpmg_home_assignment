#!/usr/bin/env python3
"""
System testing script for Medical Services ChatBot
"""

import httpx
import asyncio
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

class SystemTester:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_api_health(self) -> bool:
        """Test API health endpoint"""
        print("🔍 Testing API health...")
        try:
            response = await self.client.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                print("✅ API is healthy")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    async def test_vector_store(self) -> bool:
        """Test vector store status"""
        print("🔍 Testing vector store...")
        try:
            response = await self.client.get(f"{API_BASE_URL}/vector-store/stats")
            if response.status_code == 200:
                stats = response.json()
                if stats["status"] == "loaded":
                    print(f"✅ Vector store loaded with {stats['total_documents']} documents")
                    return True
                else:
                    print(f"❌ Vector store not loaded: {stats}")
                    return False
            else:
                print(f"❌ Vector store check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Vector store connection failed: {e}")
            return False
    
    async def test_welcome_message(self) -> bool:
        """Test welcome message endpoint"""
        print("🔍 Testing welcome message...")
        try:
            response = await self.client.get(f"{API_BASE_URL}/welcome")
            if response.status_code == 200:
                message = response.json()["message"]
                if message and len(message) > 10:
                    print("✅ Welcome message retrieved successfully")
                    return True
                else:
                    print("❌ Welcome message is empty or too short")
                    return False
            else:
                print(f"❌ Welcome message failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Welcome message failed: {e}")
            return False
    
    async def test_onboarding_chat(self) -> bool:
        """Test onboarding conversation"""
        print("🔍 Testing onboarding chat...")
        try:
            request_data = {
                "message": "שלום, שמי דן כהן",
                "user_profile": None,
                "conversation_history": [],
                "phase": "onboarding"
            }
            
            response = await self.client.post(
                f"{API_BASE_URL}/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["message"] and result["phase"] == "onboarding":
                    print("✅ Onboarding chat works")
                    print(f"   Response: {result['message'][:100]}...")
                    return True
                else:
                    print(f"❌ Unexpected onboarding response: {result}")
                    return False
            else:
                print(f"❌ Onboarding chat failed: {response.status_code}")
                try:
                    error = response.json()
                    print(f"   Error: {error}")
                except:
                    pass
                return False
        except Exception as e:
            print(f"❌ Onboarding chat error: {e}")
            return False
    
    async def test_full_onboarding_flow(self) -> Dict[str, Any]:
        """Test complete onboarding flow with long-running LLM session"""
        print("🔍 Testing full onboarding flow...")
        
        conversation_history = []
        user_profile = None
        phase = "onboarding"
        
        # Test messages in sequence for long-running LLM collection
        test_messages = [
            "שלום, שמי דן כהן",  # Should start collecting info
            "מספר הזהות שלי הוא 123456789",  # Should acknowledge and ask for more
            "אני זכר",  # Should acknowledge and ask for more
            "נולדתי ב-15/05/1985",  # Should acknowledge and ask for more
            "אני חבר במכבי",  # Should acknowledge and ask for more
            "יש לי חבילת זהב",  # Should complete collection and transition to Q&A
        ]
        
        try:
            for i, message in enumerate(test_messages):
                print(f"   Step {i+1}: {message}")
                
                request_data = {
                    "message": message,
                    "user_profile": user_profile,
                    "conversation_history": conversation_history,
                    "phase": phase
                }
                
                response = await self.client.post(
                    f"{API_BASE_URL}/chat",
                    json=request_data
                )
                
                if response.status_code != 200:
                    print(f"❌ Step {i+1} failed: {response.status_code}")
                    return {}
                
                result = response.json()
                
                # Update state
                conversation_history.append({"role": "user", "content": message, "timestamp": "2024-01-01T00:00:00"})
                conversation_history.append({"role": "assistant", "content": result["message"], "timestamp": "2024-01-01T00:00:00"})
                
                if result.get("user_profile"):
                    user_profile = result["user_profile"]
                
                phase = result["phase"]
                
                print(f"      Bot: {result['message'][:80]}...")
                print(f"      Phase: {phase}")
                
                # If we reached Q&A phase, break
                if phase == "qa":
                    print("✅ Successfully completed onboarding!")
                    return user_profile
                    
                # Small delay between messages
                await asyncio.sleep(0.5)
            
            # If we finished all messages but didn't reach Q&A phase, try one more completion message
            if phase != "qa":
                print("   Trying completion message...")
                request_data = {
                    "message": "זה הכל, יש לכם את כל המידע שלי",
                    "user_profile": user_profile,
                    "conversation_history": conversation_history,
                    "phase": phase
                }
                
                response = await self.client.post(f"{API_BASE_URL}/chat", json=request_data)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("user_profile"):
                        user_profile = result["user_profile"]
                    if result["phase"] == "qa":
                        print("✅ Successfully completed onboarding with completion message!")
                        return user_profile
            
            if phase != "qa":
                print("❌ Onboarding did not complete to Q&A phase")
            
            return user_profile or {}
            
        except Exception as e:
            print(f"❌ Full onboarding test error: {e}")
            return {}
    
    async def test_qa_chat(self, user_profile: Dict[str, Any]) -> bool:
        """Test Q&A conversation"""
        if not user_profile:
            print("⚠️  Skipping Q&A test - no user profile")
            return False
            
        print("🔍 Testing Q&A chat...")
        try:
            request_data = {
                "message": "איזה בדיקות אני זכאי בחבילת הזהב של מכבי?",
                "user_profile": user_profile,
                "conversation_history": [
                    {"role": "assistant", "content": "המידע שלך נשמר. מה תרצה לדעת?", "timestamp": "2024-01-01T00:00:00"}
                ],
                "phase": "qa"
            }
            
            response = await self.client.post(
                f"{API_BASE_URL}/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["message"] and result["phase"] == "qa":
                    print("✅ Q&A chat works")
                    print(f"   Response: {result['message'][:100]}...")
                    return True
                else:
                    print(f"❌ Unexpected Q&A response: {result}")
                    return False
            else:
                print(f"❌ Q&A chat failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Q&A chat error: {e}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all system tests"""
        print("🧪 Starting Medical Services ChatBot System Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test sequence
        tests = [
            ("API Health", self.test_api_health()),
            ("Vector Store", self.test_vector_store()),
            ("Welcome Message", self.test_welcome_message()),
            ("Basic Onboarding", self.test_onboarding_chat()),
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = await test_coro
            results.append(result)
            
            if not result:
                print(f"❌ {test_name} failed - stopping tests")
                break
        
        # Run full onboarding flow if basic tests pass
        if all(results):
            print(f"\n{'='*20} Full Onboarding Flow {'='*20}")
            user_profile = await self.test_full_onboarding_flow()
            
            if user_profile:
                print(f"\n{'='*20} Q&A Testing {'='*20}")
                qa_result = await self.test_qa_chat(user_profile)
                results.append(qa_result)
        
        # Summary
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        passed = sum(results)
        total = len(results)
        
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        print(f"⏱️  Time: {time.time() - start_time:.2f}s")
        
        if passed == total:
            print("🎉 All tests passed! System is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Please check the logs above.")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

async def main():
    """Main test function"""
    tester = SystemTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    finally:
        await tester.close()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)