import { update_header } from 'public/header.js'; // if using a separate module
import { authentication } from 'wix-members';
import wixLocation from 'wix-location';
import wixWindow from 'wix-window';

$w.onReady(async () => {
  await update_header();
  if (wixLocation.path[0] === "chat") { // Ensure the page URL contains "chat"
    $w("#model").enable(); // Disable the button   
  }
  else {
    $w("#model").disable(); // Disable the butto
  }
  $w("#dropdown").value = "default";
});

// Update the header when the user logs in
authentication.onLogin(async () => {
  await update_header();
});

authentication.onLogout(async () => {
    wixLocation.to("/about");

  });

$w("#dropdown").onChange(async (event) => {
    const chosenValue = event.target.value;
    if (chosenValue === "profile") {
        wixLocation.to("/account");
    } else if (chosenValue === "logout") {
        await wixWindow.openLightbox("Logout")
       
    }

    $w("#dropdown").value = "default";
});


