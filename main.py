# main.py
import os
import warnings

# Suppress all warnings and verbose output before any other imports
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, Button
from textual.reactive import reactive
from textual.screen import Screen
import datetime
import getpass
import asyncio
from core.compatibility_checker import CompatibilityChecker
from core.system_checker import SystemChecker

# Suppress OpenCV output after import
try:
    import cv2
    cv2.setLogLevel(0)
except:
    pass

class MainMenuScreen(Screen):
    """Main menu screen after successful verification"""
    
    def __init__(self, user_name: str):
        super().__init__()
        self.user_name = user_name
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def compose(self) -> ComposeResult:
        """Create the main menu layout"""
        yield Header()
        yield Container(
            Vertical(
                Static("🤟 Sign Language Keyboard", id="main-title"),
                Static(f"Main Menu - {self.user_name} | {self.current_time}", id="main-subtitle"),
                Button("Start", id="start-btn", variant="success"),
                Button("Train", id="train-btn", variant="primary"),
                Button("Settings", id="settings-btn", variant="default"),
                Button("Exit", id="main-exit-btn", variant="error"),
                id="main-menu-content"
            ),
            id="main-menu-container"
        )
        yield Footer()
    
    def on_resize(self, event) -> None:
        """Handle terminal resize events for main menu"""
        size = self.size
        
        try:
            # Adjust title based on terminal width
            title = self.query_one("#main-title", Static)
            
            if size.width < 50:
                title.update("🤟 Sign Lang. KB")
            elif size.width < 70:
                title.update("🤟 Sign Lang. Keyboard")
            elif size.width < 100:
                title.update("🤟 Sign Language Keyboard")
            else:
                title.update("🤟 Sign Language Keyboard")
            
            # Adjust subtitle for very small screens
            subtitle = self.query_one("#main-subtitle", Static)
            if size.width < 60:
                subtitle.update(f"Menu - {self.user_name}")
            else:
                subtitle.update(f"Main Menu - {self.user_name} | {self.current_time}")
                
        except Exception:
            # Ignore errors if widgets aren't ready yet
            pass
    
    def on_mount(self) -> None:
        """Called when the screen starts"""
        self.on_resize(None)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks in main menu"""
        if event.button.id == "start-btn":
            # TODO: Implement start functionality
            self.notify("Start feature not yet implemented", title="Coming Soon")
            
        elif event.button.id == "train-btn":
            # TODO: Implement training functionality
            self.notify("Training feature not yet implemented", title="Coming Soon")
            
        elif event.button.id == "settings-btn":
            # TODO: Implement settings functionality
            self.notify("Settings feature not yet implemented", title="Coming Soon")
            
        elif event.button.id == "main-exit-btn":
            self.app.exit()

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
        self.system_checker = SystemChecker()
        self.verification_running = False
        self.current_worker = None
        self.verification_passed = False
        
    @property
    def sub_title(self) -> str:
        return self._sub_title
    
    @sub_title.setter
    def sub_title(self, value: str) -> None:
        self._sub_title = value
    
    def compose(self) -> ComposeResult:
        """Create the basic UI layout with responsive design"""
        yield Header()
        # Verification Layout
        yield Container(
            Vertical(
                Static("🤟 Sign Language Keyboard", id="title"),
                Static(f"Welcome back, {self.current_user}!", id="welcome"),
                Static("Current Status: Ready to Initialize", id="status"),
                Button("System Verification", id="verify-btn", variant="success"),
                Button("Exit", id="exit-btn", variant="error"),
                id="main-content"
            ),
            id="main-container"
        )
        yield Footer()
    
    def on_resize(self, event) -> None:
        """Handle terminal resize events to adjust layout"""
        # Get terminal size
        size = self.size
        
        try:
            # Adjust title based on terminal width
            title = self.query_one("#title", Static)
            
            if size.width < 50:
                title.update("🤟 Sign Lang. KB")
            elif size.width < 70:
                title.update("🤟 Sign Lang. Keyboard")
            elif size.width < 100:
                title.update("🤟 Sign Language Keyboard")
            else:
                title.update("🤟 Sign Language Keyboard")
            
            # Adjust welcome message for very small screens
            welcome = self.query_one("#welcome", Static)
            if size.width < 50:
                welcome.update(f"Welcome, {self.current_user}!")
            else:
                welcome.update(f"Welcome back, {self.current_user}!")
                
        except Exception:
            # Ignore errors if widgets aren't ready yet
            pass
    
    def on_mount(self) -> None:
        """Called when the app starts - set initial responsive state"""
        # Initial responsive adjustment
        self.on_resize(None)

    async def run_progressive_checks(self):
        """Run comprehensive system checks with progressive updates"""
        status_widget = self.query_one("#status", Static)
        
        try:
            # Step 1: Initialize
            status_widget.update("Status: [1/8] Initializing comprehensive system scan...")
            await asyncio.sleep(0.5)
            
            # Step 2: Check Python version and libraries (compatibility)
            status_widget.update("Status: [2/8] Checking Python compatibility...")
            python_ok, python_msg = self.checker.check_python_version()
            await asyncio.sleep(0.5)
            
            # Step 3: Check libraries
            status_widget.update("Status: [3/8] Checking required libraries...")
            libraries_ok, library_messages = self.checker.check_libraries()
            await asyncio.sleep(0.5)
            
            # Step 4: Check hardware components
            status_widget.update("Status: [4/8] Scanning hardware components...")
            await asyncio.sleep(0.5)
            
            # Step 5: Check display configuration
            status_widget.update("Status: [5/8] Detecting display configuration...")
            await asyncio.sleep(0.5)
            
            # Step 6: Check camera availability
            status_widget.update("Status: [6/8] Testing camera availability...")
            await asyncio.sleep(0.5)
            
            # Step 7: Generate comprehensive report
            status_widget.update("Status: [7/8] Generating comprehensive system report...")
            results, report_path = self.system_checker.run_system_checks()
            await asyncio.sleep(0.5)
            
            # Step 8: Final status
            status_widget.update("Status: [8/8] Finalizing system analysis...")
            await asyncio.sleep(0.5)
            
            # Determine overall status (compatibility + system report)
            compatibility_ok = python_ok and libraries_ok
            system_report_ok = results['status'] in ['success', 'completed_with_errors']
            
            # Update final status based on all results
            if compatibility_ok and results['status'] == 'success':
                status_widget.update("Status: ✅ All systems compatible and ready!")
                self.verification_passed = True
                # Immediately open main menu screen
                self.open_main_menu()
            elif compatibility_ok and results['status'] == 'completed_with_errors':
                error_count = len(results.get('errors', []))
                status_widget.update(f"Status: ⚠️ System compatible with {error_count} minor warnings")
                self.verification_passed = True
                # Immediately open main menu screen
                self.open_main_menu()
            elif not compatibility_ok and system_report_ok:
                status_widget.update("Status: ❌ Compatibility issues detected. Please check requirements")
            else:
                status_widget.update("Status: ❌ Multiple system issues found. Please review setup")
                
        except Exception as e:
            status_widget.update(f"Status: ❌ System check failed: {str(e)}")
        
        finally:
            # Reset button state when verification completes
            self.verification_running = False
            if not self.verification_passed:  # Only reset to original state if verification didn't pass
                try:
                    verify_btn = self.query_one("#verify-btn", Button)
                    verify_btn.label = "System Verification"
                    verify_btn.variant = "success"
                except:
                    pass  # Widget might not exist
            self.current_worker = None
    
    def open_main_menu(self) -> None:
        """Open the main menu screen after successful verification"""
        main_menu = MainMenuScreen(self.current_user)
        self.push_screen(main_menu)
    
    def start_verification(self) -> None:
        """Start the verification process"""
        self.verification_running = True
        verify_btn = self.query_one("#verify-btn", Button)
        verify_btn.label = "Stop"
        verify_btn.variant = "primary"
        
        # Reset verification state if restarting
        if self.verification_passed:
            self.verification_passed = False
            # Reset status message
            status = self.query_one("#status", Static)
            status.update("Current Status: Restarting verification...")
        
        # Start the verification worker
        self.current_worker = self.run_worker(self.run_progressive_checks(), exclusive=True)
    
    def stop_verification(self) -> None:
        """Stop the verification process"""
        if self.current_worker:
            self.current_worker.cancel()
        
        self.verification_running = False
        verify_btn = self.query_one("#verify-btn", Button)
        verify_btn.label = "System Verification"
        verify_btn.variant = "success"
        
        # Update status
        status = self.query_one("#status", Static)
        status.update("Current Status: Verification Stopped")
        
        self.current_worker = None
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        # Clear any text selection/highlighting
        self.clear_selection()
        
        # Verification page button handling
        if event.button.id == "verify-btn":
            button_text = event.button.label
            if button_text == "Stop":
                # Currently running, stop it
                self.stop_verification()
            elif button_text in ["System Verification", "Start Verification"]:
                # Start or restart verification
                self.start_verification()
            
        elif event.button.id == "exit-btn":
            self.exit()
    
    def clear_selection(self) -> None:
        """Clear any text selection/highlighting in the terminal"""
        try:
            # Force a refresh of the button focus to clear highlighting
            if hasattr(self, 'focused') and self.focused:
                self.focused.blur()
            # Force a screen refresh
            self.refresh()
        except Exception:
            # Ignore any errors during selection clearing
            pass

if __name__ == "__main__":
    app = SignLanguageKeyboardApp()
    app.run()
