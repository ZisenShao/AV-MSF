document.addEventListener("DOMContentLoaded", () => {
  const cards = Array.from(document.querySelectorAll(".object-card"));
  const stageLabel = document.getElementById("demo-stage-label");
  const previousButton = document.getElementById("carousel-prev");
  const nextButton = document.getElementById("carousel-next");
  const modeButtons = Array.from(
    document.querySelectorAll(".demo-switcher-button"),
  );

  let activeIndex = 0;
  let activeMode = "Impact Sound Rendering";

  function updateStageLabel() {
    const activeCard = cards[activeIndex];
    stageLabel.textContent =
      `${activeCard.dataset.name} · ${activeMode} placeholder`;
  }

  function selectCard(index) {
    activeIndex = (index + cards.length) % cards.length;

    cards.forEach((card, cardIndex) => {
      card.classList.toggle("active", cardIndex === activeIndex);
    });

    const activeCard = cards[activeIndex];
    updateStageLabel();
    activeCard.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "center",
    });
  }

  cards.forEach((card, index) => {
    card.addEventListener("click", () => selectCard(index));
  });

  modeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activeMode = button.dataset.mode;

      modeButtons.forEach((modeButton) => {
        const isActive = modeButton === button;
        modeButton.classList.toggle("active", isActive);
        modeButton.setAttribute("aria-pressed", String(isActive));
      });

      updateStageLabel();
    });
  });

  previousButton.addEventListener("click", () => selectCard(activeIndex - 1));
  nextButton.addEventListener("click", () => selectCard(activeIndex + 1));
});
