const resultTabs = [...document.querySelectorAll('.results-tabs [role="tab"]')];

function selectResultTab(selectedTab) {
  resultTabs.forEach((tab) => {
    const selected = tab === selectedTab;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
  });
}

resultTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectResultTab(tab));
  tab.addEventListener('keydown', (event) => {
    let nextIndex;
    switch (event.key) {
      case 'ArrowRight': nextIndex = (index + 1) % resultTabs.length; break;
      case 'ArrowLeft': nextIndex = (index - 1 + resultTabs.length) % resultTabs.length; break;
      case 'Home': nextIndex = 0; break;
      case 'End': nextIndex = resultTabs.length - 1; break;
      default: return;
    }
    event.preventDefault();
    selectResultTab(resultTabs[nextIndex]);
    resultTabs[nextIndex].focus();
  });
});
