document.querySelectorAll("[data-reveal-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.revealTarget);
    if (!target) return;
    target.classList.toggle("is-visible");
    button.textContent = target.classList.contains("is-visible")
      ? "Hide answer"
      : "Show answer";
  });
});
