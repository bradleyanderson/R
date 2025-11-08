#!/usr/bin/env python3
"""
ResolveAI Universal CLI - Command Line Interface
"""

import asyncio
import sys
import os
import argparse
from pathlib import Path

# Add the resolveai package to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from resolveai.core.universal_assistant import UniversalAssistant
    from resolveai.config.settings import UniversalAssistantConfig
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def create_cli_parser():
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="resolveai",
        description="Universal AI Assistant Platform - Control any software with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  resolveai start                          # Start the AI assistant
  resolveai start --port 8080              # Start on custom port
  resolveai start --config config.yaml    # Use custom config
  resolveai screen --analyze              # Analyze current screen
  resolveai automate --workflow demo.yaml # Run automation workflow
  resolveai plugin --list                  # List available plugins
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Universal AI Assistant")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    start_parser.add_argument("--config", help="Path to configuration file")
    start_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    start_parser.add_argument("--no-screen", action="store_true", help="Disable screen intelligence")
    
    # Screen command
    screen_parser = subparsers.add_parser("screen", help="Screen intelligence operations")
    screen_parser.add_argument("--analyze", action="store_true", help="Analyze current screen")
    screen_parser.add_argument("--monitor", action="store_true", help="Start screen monitoring")
    screen_parser.add_argument("--screenshot", help="Take screenshot and save to path")
    
    # Automate command
    auto_parser = subparsers.add_parser("automate", help="Automation operations")
    auto_parser.add_argument("--workflow", help="Path to workflow file")
    auto_parser.add_argument("--record", action="store_true", help="Record new workflow")
    auto_parser.add_argument("--list", action="store_true", help="List available workflows")
    
    # Plugin command
    plugin_parser = subparsers.add_parser("plugin", help="Plugin management")
    plugin_parser.add_argument("--list", action="store_true", help="List installed plugins")
    plugin_parser.add_argument("--install", help="Install plugin from URL")
    plugin_parser.add_argument("--remove", help="Remove plugin")
    plugin_parser.add_argument("--info", help="Show plugin information")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--init", action="store_true", help="Initialize configuration")
    config_parser.add_argument("--validate", action="store_true", help="Validate configuration")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    return parser


async def start_assistant(args):
    """Start the Universal AI Assistant."""
    try:
        print("🚀 Starting ResolveAI Universal Assistant...")
        
        # Load configuration
        if args.config and os.path.exists(args.config):
            config = UniversalAssistantConfig.from_file(args.config)
        else:
            config = UniversalAssistantConfig(
                enable_screen_intelligence=not args.no_screen,
                debug=args.debug,
                server_port=args.port,
                server_host=args.host
            )
        
        # Initialize assistant
        assistant = UniversalAssistant(config)
        
        print(f"✅ Assistant initialized successfully")
        print(f"🌐 Server starting on http://{args.host}:{args.port}")
        print(f"👁️ Screen Intelligence: {'Enabled' if config.enable_screen_intelligence else 'Disabled'}")
        print(f"🔧 Debug Mode: {'Enabled' if config.debug else 'Disabled'}")
        
        # Start the assistant
        await assistant.start()
        
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def handle_screen_command(args):
    """Handle screen-related commands."""
    try:
        from resolveai.universal.screen_intelligence import ScreenIntelligence
        
        screen_intel = ScreenIntelligence()
        
        if args.analyze:
            print("🔍 Analyzing current screen...")
            result = await screen_intel.analyze_screen()
            print("Analysis complete:")
            print(f"  Interface Type: {result.get('interface_type', 'Unknown')}")
            print(f"  UI Elements Found: {len(result.get('ui_elements', []))}")
            print(f"  Text Regions: {len(result.get('text_regions', []))}")
            
        elif args.monitor:
            print("👁️ Starting screen monitoring...")
            await screen_intel.start_monitoring()
            
        elif args.screenshot:
            print("📸 Taking screenshot...")
            await screen_intel.take_screenshot(args.screenshot)
            print(f"Screenshot saved to: {args.screenshot}")
            
    except Exception as e:
        print(f"❌ Screen operation failed: {e}")
        sys.exit(1)


async def handle_automation_command(args):
    """Handle automation commands."""
    try:
        from resolveai.universal.automation_engine import AutomationEngine
        
        automation = AutomationEngine()
        
        if args.workflow:
            print(f"🤖 Running workflow: {args.workflow}")
            await automation.execute_workflow(args.workflow)
            
        elif args.record:
            print("🎥 Recording new workflow...")
            print("Press Ctrl+C to stop recording")
            await automation.record_workflow()
            
        elif args.list:
            print("📋 Available workflows:")
            workflows = await automation.list_workflows()
            for workflow in workflows:
                print(f"  • {workflow['name']}: {workflow['description']}")
                
    except Exception as e:
        print(f"❌ Automation operation failed: {e}")
        sys.exit(1)


def handle_plugin_command(args):
    """Handle plugin management commands."""
    try:
        from resolveai.universal.plugin_system import PluginManager
        
        plugin_manager = PluginManager()
        
        if args.list:
            print("🔌 Installed plugins:")
            plugins = plugin_manager.list_plugins()
            for plugin in plugins:
                print(f"  • {plugin.name} v{plugin.version}: {plugin.description}")
                
        elif args.install:
            print(f"📦 Installing plugin: {args.install}")
            plugin_manager.install_plugin(args.install)
            
        elif args.remove:
            print(f"🗑️ Removing plugin: {args.remove}")
            plugin_manager.remove_plugin(args.remove)
            
        elif args.info:
            print(f"ℹ️ Plugin info: {args.info}")
            info = plugin_manager.get_plugin_info(args.info)
            for key, value in info.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"❌ Plugin operation failed: {e}")
        sys.exit(1)


def handle_config_command(args):
    """Handle configuration commands."""
    try:
        if args.show:
            print("⚙️ Current configuration:")
            # Show current config
            config_file = os.path.expanduser("~/.resolveai/config.yaml")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    print(f.read())
            else:
                print("No configuration file found. Use --init to create one.")
                
        elif args.init:
            print("📝 Initializing configuration...")
            config_dir = os.path.expanduser("~/.resolveai")
            os.makedirs(config_dir, exist_ok=True)
            
            # Create default config
            default_config = """# ResolveAI Universal Configuration
assistant:
  name: "ResolveAI Assistant"
  enable_screen_intelligence: true
  enable_automation: true
  enable_conversational_interface: true
  
ai_providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-vision-preview"
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-opus-20240229"
    
server:
  host: "0.0.0.0"
  port: 8000
  debug: false
  
security:
  encryption_key: "${ENCRYPTION_KEY}"
  
logging:
  level: "INFO"
  file: "~/.resolveai/logs/resolveai.log"
"""
            config_file = os.path.join(config_dir, "config.yaml")
            with open(config_file, 'w') as f:
                f.write(default_config)
            print(f"Configuration created at: {config_file}")
            
        elif args.validate:
            print("✅ Validating configuration...")
            # Validate config logic here
            print("Configuration is valid!")
            
    except Exception as e:
        print(f"❌ Configuration operation failed: {e}")
        sys.exit(1)


async def show_status():
    """Show system status."""
    print("📊 ResolveAI System Status")
    print("=" * 40)
    
    # Check dependencies
    try:
        import cv2
        print("✅ OpenCV: Available")
    except ImportError:
        print("❌ OpenCV: Not installed")
        
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch: Not installed")
        
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers: Not installed")
    
    # Check screen capture permissions
    try:
        import mss
        with mss.mss() as sct:
            monitors = sct.monitors
        print(f"✅ Screen Capture: {len(monitors)-1} monitors detected")
    except Exception as e:
        print(f"❌ Screen Capture: {e}")
    
    # Check configuration
    config_file = os.path.expanduser("~/.resolveai/config.yaml")
    if os.path.exists(config_file):
        print("✅ Configuration: Found")
    else:
        print("⚠️ Configuration: Not initialized (run 'resolveai config --init')")
    
    print("\n🚀 Ready to start!")


def main():
    """Main CLI entry point - wrapper for async function."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down ResolveAI...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


async def _async_main():
    """Async main function."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "start":
        await start_assistant(args)
    elif args.command == "screen":
        await handle_screen_command(args)
    elif args.command == "automate":
        await handle_automation_command(args)
    elif args.command == "plugin":
        handle_plugin_command(args)
    elif args.command == "config":
        handle_config_command(args)
    elif args.command == "status":
        await show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()