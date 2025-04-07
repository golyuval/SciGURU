// In a common module (e.g. public/headerUtils.js) or directly in masterPage.js
import { currentMember, authentication } from 'wix-members';

// Shared function to update the header profile icon
export async function update_header() {
  
  const member = await currentMember.getMember();
  
  if (member && member.profile) {
    
    if (member.profile.profilePhoto && member.profile.profilePhoto.url) {
      $w("#profileIcon").src = member.profile.profilePhoto.url;
      $w("#profileIcon").show();
    }

    $w("#loginButton").disable();
  }
  
  else {
    $w("#dropdown").disable();
    $w("#loginButton").enable();
  }
}


