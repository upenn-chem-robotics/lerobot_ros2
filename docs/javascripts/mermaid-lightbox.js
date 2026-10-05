(() => {
  const enhance = () => {
    document.querySelectorAll(".mermaid").forEach((diagram, index) => {
      if (diagram.dataset.lightboxReady === "true") return;
      const source = diagram.querySelector("svg");
      if (!source) return;

      diagram.dataset.lightboxReady = "true";
      diagram.tabIndex = 0;
      diagram.setAttribute("role", "button");
      diagram.setAttribute("aria-label", `Enlarge diagram ${index + 1}`);

      const open = () => {
        const currentSource = diagram.querySelector("svg");
        if (!currentSource) return;

        const dialog = document.createElement("dialog");
        dialog.className = "mermaid-lightbox";
        dialog.setAttribute("aria-label", "Enlarged diagram");

        const toolbar = document.createElement("div");
        toolbar.className = "mermaid-lightbox__toolbar";

        const closeButton = document.createElement("button");
        closeButton.className = "mermaid-lightbox__close";
        closeButton.type = "button";
        closeButton.textContent = "Close";
        closeButton.setAttribute("aria-label", "Close enlarged diagram");

        const viewport = document.createElement("div");
        viewport.className = "mermaid-lightbox__viewport";

        const clone = currentSource.cloneNode(true);
        clone.removeAttribute("style");
        viewport.appendChild(clone);
        toolbar.appendChild(closeButton);
        dialog.append(toolbar, viewport);
        document.body.appendChild(dialog);

        closeButton.addEventListener("click", () => dialog.close());
        dialog.addEventListener("click", event => {
          const box = dialog.getBoundingClientRect();
          const inside = event.clientX >= box.left && event.clientX <= box.right &&
                         event.clientY >= box.top && event.clientY <= box.bottom;
          if (!inside) dialog.close();
        });
        dialog.addEventListener("close", () => {
          dialog.remove();
          diagram.focus();
        }, { once: true });

        dialog.showModal();
      };

      diagram.addEventListener("click", open);
      diagram.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          open();
        }
      });
    });
  };

  const rescan = () => {
    enhance();
    requestAnimationFrame(enhance);
    window.setTimeout(enhance, 100);
    window.setTimeout(enhance, 500);
  };

  document.addEventListener("DOMContentLoaded", rescan);
  window.addEventListener("load", rescan);
  if (typeof document$ !== "undefined") document$.subscribe(rescan);

  new MutationObserver(enhance).observe(document.documentElement, {
    childList: true,
    subtree: true
  });

  rescan();
})();
