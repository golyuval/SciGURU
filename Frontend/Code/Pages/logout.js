import { authentication } from 'wix-members';
import wixLocation from 'wix-location';
import wixWindow from 'wix-window';

$w.onReady(() => {
  $w("#bye").onClick(async () => {
    try {
      // Log the user out first
      await authentication.logout();

      // Close the lightbox
      wixWindow.lightbox.close();
      
    } catch (err) {
      console.error("Error during logout:", err);
    }
  });
});
