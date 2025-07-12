# main.py
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, Button
from textual.reactive import reactive
import datetime
import getpass
import asyncio
from core.compatibility_checker import CompatibilityChecker

class SignLanguageKeyboardApp(App):
    """Sign Language Keyboard Terminal Application"""
    
    TITLE = "Sign Language Keyboard"
    CSS_PATH = "styles.css"
    
    def __init__(self):
        super().__init__()
        self.current_user = getpass.getuser()  # Gets the actual logged-in user
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._sub_title = f"Welcome {self.current_user} - {self.current_time}"
        self.checker = CompatibilityChecker()
        
    @property
    def sub_title(self) -> str:
        return self._sub_title
    
    @sub_title.setter
    def sub_title(self, value: str) -> None:
        self._sub_title = value
    
    def compose(self) -> ComposeResult:
        """Create the basic UI layout"""
        yield Header()
        yield Container(
            Vertical(
                Static("🤟 Sign Language Keyboard", id="title"),
                Static(f"Welcome back, {self.current_user}!", id="welcome"),
                Static("Current Status: Ready to Initialize", id="status"),
                Button("Start System Checks", id="start-btn", variant="primary"),
                Button("Exit", id="exit-btn", variant="error"),
                id="main-content"
            ),
            id="main-container"
        )
        yield Footer()
    
    async def run_progressive_checks(self):
        """Run compatibility checks with progressive updates"""
        status_widget = self.query_one("#status", Static)
        
        # Step 1: Initialize
        status_widget.update("Status: [1/5] Initializing system checks...")
        await asyncio.sleep(0.5)
        
        # Step 2: Check Python version
        status_widget.update("Status: [2/5] Checking Python version...")
        python_ok, python_msg = self.checker.check_python_version()
        await asyncio.sleep(0.5)
        
        # Step 3: Check libraries
        status_widget.update("Status: [3/5] Checking required libraries...")
        libraries_ok, library_messages = self.checker.check_libraries()
        await asyncio.sleep(0.5)
        
        # Step 4: Get system specs
        status_widget.update("Status: [4/5] Gathering system specifications...")
        system_specs = self.checker.get_system_specs()
        await asyncio.sleep(0.5)
        
        # Step 5: Final status
        status_widget.update("Status: [5/5] Finalizing compatibility report...")
        overall_status = python_ok and libraries_ok
        await asyncio.sleep(0.5)
        
        # Update final status
        if overall_status:
            status_widget.update("Status: ✅ System is compatible! Ready to proceed.")
        else:
            status_widget.update("Status: ❌ Compatibility issues detected. Please install missing components.")
    
    def on_mount(self) -> None:
        """Called when the app starts"""
        pass
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "start-btn":
            self.run_worker(self.run_progressive_checks(), exclusive=True)
            
        elif event.button.id == "exit-btn":
            self.exit()

if __name__ == "__main__":
    app = SignLanguageKeyboardApp()
    app.run()