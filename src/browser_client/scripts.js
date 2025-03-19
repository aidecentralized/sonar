// Simple script to set (and potentially update) user stats
// For a real app, you'd fetch these from a backend or database.

let monthlyCount = 20000;
let dailyCount = 6000;

document.addEventListener("DOMContentLoaded", () => {
  // Set initial counts
  const monthlyUsersEl = document.getElementById("monthlyUsers");
  const dailyUsersEl = document.getElementById("dailyUsers");

  if (monthlyUsersEl) monthlyUsersEl.textContent = `${monthlyCount}k+`;
  if (dailyUsersEl) dailyUsersEl.textContent = `${dailyCount}k+`;
});

// In a real scenario, you could define a function to increment or update counts, e.g.:
// function addNewUser() {
//   monthlyCount++;
//   dailyCount++;
//   document.getElementById("monthlyUsers").textContent = `${monthlyCount}+`;
//   document.getElementById("dailyUsers").textContent = `${dailyCount}+`;
// }
