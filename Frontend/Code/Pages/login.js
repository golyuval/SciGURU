import { authentication, currentMember } from 'wix-members';
import wixLocation from 'wix-location';

$w.onReady(async () => {
  
  // ------------------- already logged in ---------------------------------------------------------

  let member = await currentMember.getMember();
  
  if (member) {
    $w("#profileIcon").src = member.profile.profilePhoto.url;
    $w("#profileIcon").show();
    wixLocation.to("/chat");
  } 
  
  else {
    $w("#dropdown").disable();
  }

  // ------------------- log in ---------------------------------------------------------

  $w("#login").onClick(async () => {
    
    $w("#login").disable();
    $w("#signup").disable();
    try {

      await authentication.promptLogin({ mode: "login" });
      member = await currentMember.getMember();
      
      // Proceed only if member data is available
      if (member) {
        $w("#profileIcon").src = member.profile.profilePhoto.url;
        $w("#profileIcon").show();

        await new Promise(resolve => setTimeout(resolve, 300));
        wixLocation.to("/chat");
      } 
      else {
        console.error("Member data not found after login.");
      }
    } 
    catch (err) {
      console.error("Login error:", err);
    } 
    finally {
      $w("#login").enable();
      $w("#signup").enable();
    }
  });

  // ------------------- sign in ---------------------------------------------------------

  $w("#signup").onClick(async () => 
    {
    $w("#login").disable();
    $w("#signup").disable();
    try {

      await authentication.promptLogin({ mode: "signup" });
      member = await currentMember.getMember();

      if (member) {
        $w("#profileIcon").src = member.profile.profilePhoto.url;
        $w("#profileIcon").show();
      }
    } 
    catch (err) {
      console.error("Sign up error:", err);
    } 
    finally {
      $w("#login").enable();
      $w("#signup").enable();
    }
  });
});
