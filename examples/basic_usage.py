#!/usr/bin/env python3
"""
ResolveAI Universal Platform - Basic Usage Examples

This file demonstrates the fundamental capabilities of ResolveAI.
Run these examples to get familiar with the platform.
"""

import asyncio
import os
from pathlib import Path

# Add the resolveai package to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from resolveai.core.universal_assistant import UniversalAssistant
from resolveai.config.settings import UniversalAssistantConfig


async def example_1_basic_startup():
    """Example 1: Basic startup and initialization."""
    print("🚀 Example 1: Basic Startup")
    print("-" * 40)
    
    try:
        # Create basic configuration
        config = UniversalAssistantConfig(
            enable_screen_intelligence=True,
            enable_automation=True,
            enable_conversational_interface=True,
            debug=True
        )
        
        # Initialize assistant
        assistant = UniversalAssistant(config)
        
        # Start the assistant
        await assistant.start()
        
        print("✅ Assistant started successfully!")
        print(f"🌐 Web interface available at: http://localhost:{config.server_port}")
        
        return assistant
        
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        return None


async def example_2_screen_analysis(assistant):
    """Example 2: Analyze the current screen."""
    print("\n🔍 Example 2: Screen Analysis")
    print("-" * 40)
    
    try:
        # Analyze current screen
        analysis = await assistant.analyze_screen()
        
        print("📊 Screen Analysis Results:")
        print(f"  • Interface Type: {analysis.get('interface_type', 'Unknown')}")
        print(f"  • UI Elements Found: {len(analysis.get('ui_elements', []))}")
        print(f"  • Text Regions: {len(analysis.get('text_regions', []))}")
        print(f"  • Confidence: {analysis.get('confidence', 0):.2f}")
        
        # Show detected elements
        if analysis.get('ui_elements'):
            print("\n🎯 Detected UI Elements:")
            for i, element in enumerate(analysis['ui_elements'][:5]):  # Show first 5
                print(f"  {i+1}. {element.get('type', 'Unknown')} - {element.get('text', 'No text')}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Screen analysis failed: {e}")
        return None


async def example_3_natural_language_control(assistant):
    """Example 3: Control software with natural language."""
    print("\n💬 Example 3: Natural Language Control")
    print("-" * 40)
    
    try:
        # Example commands
        commands = [
            "What applications are currently open?",
            "Find all clickable buttons on the screen",
            "What text is visible in the current window?"
        ]
        
        for i, command in enumerate(commands, 1):
            print(f"\n🗣️ Command {i}: {command}")
            
            # Process the command
            response = await assistant.process_request({
                "type": "conversational",
                "input": command
            })
            
            print(f"🤖 Response: {response.get('response', 'No response')}")
            
    except Exception as e:
        print(f"❌ Natural language control failed: {e}")


async def example_4_automation_workflow(assistant):
    """Example 4: Create and execute a workflow."""
    print("\n⚙️ Example 4: Automation Workflow")
    print("-" * 40)
    
    try:
        # Define a simple workflow
        workflow = {
            "name": "Screen Capture Workflow",
            "description": "Take screenshot and analyze it",
            "steps": [
                {
                    "action": "capture_screen",
                    "params": {
                        "save_to": "workflow_screenshot.png"
                    }
                },
                {
                    "action": "analyze_screen",
                    "params": {
                        "screenshot_path": "workflow_screenshot.png"
                    }
                },
                {
                    "action": "log_results",
                    "params": {
                        "output_file": "workflow_results.txt"
                    }
                }
            ]
        }
        
        print("📋 Executing workflow:")
        print(f"  • Name: {workflow['name']}")
        print(f"  • Steps: {len(workflow['steps'])}")
        
        # Execute the workflow
        result = await assistant.execute_workflow(workflow)
        
        print(f"✅ Workflow completed: {result.get('status', 'Unknown')}")
        
        if result.get('output_files'):
            print("📁 Generated files:")
            for file in result['output_files']:
                print(f"  • {file}")
                
    except Exception as e:
        print(f"❌ Automation workflow failed: {e}")


async def example_5_plugin_system(assistant):
    """Example 5: Explore the plugin system."""
    print("\n🔌 Example 5: Plugin System")
    print("-" * 40)
    
    try:
        # List available plugins
        plugins = await assistant.list_plugins()
        
        print(f"📦 Available plugins: {len(plugins)}")
        for plugin in plugins:
            print(f"  • {plugin.get('name', 'Unknown')} v{plugin.get('version', '0.0.0')}")
            print(f"    {plugin.get('description', 'No description')}")
        
        # Show plugin capabilities
        if plugins:
            print(f"\n🎯 Plugin capabilities:")
            for plugin in plugins[:3]:  # Show first 3
                capabilities = plugin.get('capabilities', [])
                print(f"  {plugin.get('name', 'Unknown')}: {', '.join(capabilities)}")
                
    except Exception as e:
        print(f"❌ Plugin system exploration failed: {e}")


async def example_6_learning_and_adaptation(assistant):
    """Example 6: Demonstrate learning capabilities."""
    print("\n🧠 Example 6: Learning and Adaptation")
    print("-" * 40)
    
    try:
        # Get learning status
        learning_status = await assistant.get_learning_status()
        
        print("📊 Learning System Status:")
        print(f"  • Learning Enabled: {learning_status.get('enabled', False)}")
        print(f"  • Patterns Learned: {learning_status.get('patterns_count', 0)}")
        print(f"  • User Preferences: {learning_status.get('preferences_count', 0)}")
        
        # Show recent learning activities
        if learning_status.get('recent_activities'):
            print("\n📈 Recent Learning Activities:")
            for activity in learning_status['recent_activities'][:3]:
                print(f"  • {activity.get('type', 'Unknown')}: {activity.get('description', 'No description')}")
        
        # Demonstrate preference learning
        print("\n🎯 Teaching a preference...")
        await assistant.teach_preference({
            "context": "file_operations",
            "preference": "always_ask_before_deleting",
            "confidence": 0.9
        })
        
        print("✅ Preference learned!")
        
    except Exception as e:
        print(f"❌ Learning demonstration failed: {e}")


async def example_7_multi_modal_processing(assistant):
    """Example 7: Multi-modal data processing."""
    print("\n🎨 Example 7: Multi-Modal Processing")
    print("-" * 40)
    
    try:
        # Test different data types
        test_files = [
            "examples/test_image.png",
            "examples/test_audio.mp3",
            "examples/test_document.pdf"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"\n📁 Processing: {file_path}")
                
                # Process the file
                result = await assistant.process_file(file_path)
                
                print(f"  • Type: {result.get('file_type', 'Unknown')}")
                print(f"  • Size: {result.get('file_size', 0)} bytes")
                print(f"  • Processing Time: {result.get('processing_time', 0):.2f}s")
                
                if result.get('extracted_content'):
                    content_preview = result['extracted_content'][:100] + "..." if len(result['extracted_content']) > 100 else result['extracted_content']
                    print(f"  • Content Preview: {content_preview}")
            else:
                print(f"⚠️ Test file not found: {file_path}")
        
        # Test cross-modal analysis
        print("\n🔗 Cross-Modal Analysis:")
        analysis = await assistant.analyze_relationships(test_files)
        
        print(f"  • Relationships Found: {len(analysis.get('relationships', []))}")
        for relationship in analysis.get('relationships', []):
            print(f"    {relationship.get('type', 'Unknown')}: {relationship.get('description', 'No description')}")
            
    except Exception as e:
        print(f"❌ Multi-modal processing failed: {e}")


async def main():
    """Run all examples."""
    print("🎯 ResolveAI Universal Platform - Basic Usage Examples")
    print("=" * 60)
    print("This script demonstrates the core capabilities of ResolveAI.")
    print("Make sure you have configured your API keys first!")
    print("=" * 60)
    
    # Check configuration
    config_file = os.path.expanduser("~/.resolveai/config.yaml")
    if not os.path.exists(config_file):
        print("⚠️ Configuration not found. Please run: resolveai config --init")
        return
    
    # Start the assistant
    assistant = await example_1_basic_startup()
    if not assistant:
        print("❌ Failed to start assistant. Exiting.")
        return
    
    try:
        # Run examples
        await example_2_screen_analysis(assistant)
        await example_3_natural_language_control(assistant)
        await example_4_automation_workflow(assistant)
        await example_5_plugin_system(assistant)
        await example_6_learning_and_adaptation(assistant)
        await example_7_multi_modal_processing(assistant)
        
        print("\n🎉 All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("  • Open http://localhost:8000 for the web interface")
        print("  • Try the CLI: resolveai --help")
        print("  • Read the documentation: docs/")
        print("  • Join the community: discord.gg/resolveai")
        
    except KeyboardInterrupt:
        print("\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        # Clean up
        if assistant:
            await assistant.stop()
            print("\n🛑 Assistant stopped")


if __name__ == "__main__":
    asyncio.run(main())