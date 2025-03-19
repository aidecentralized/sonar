document.addEventListener("DOMContentLoaded", () => {
    const slides = document.querySelectorAll(".slide");
    const dots = document.querySelectorAll(".dot-indicator ul li");
    let currentIndex = 0;
    let isTransitioning = false;
    const throttleDuration = 1200; // Wait 1 second before detecting new scroll events
  
    slides[currentIndex].classList.add("active");
    updateDots(currentIndex);
  
    window.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        if (isTransitioning) return;
        isTransitioning = true;
        showNextSlide();
        setTimeout(() => {
          isTransitioning = false;
        }, throttleDuration);
      },
      { passive: false }
    );
  
    function showNextSlide() {
      slides[currentIndex].classList.remove("active");
      currentIndex = (currentIndex + 1) % slides.length;
      slides[currentIndex].classList.add("active");
      updateDots(currentIndex);
    }
  
    // Allow dot-click navigation
    dots.forEach((dot, index) => {
      dot.addEventListener("click", () => {
        if (isTransitioning || index === currentIndex) return;
        slides[currentIndex].classList.remove("active");
        currentIndex = index;
        slides[currentIndex].classList.add("active");
        updateDots(currentIndex);
      });
    });
  
    function updateDots(index) {
      dots.forEach((dot) => dot.classList.remove("active"));
      if (dots[index]) {
        dots[index].classList.add("active");
      }
    }
  });
  