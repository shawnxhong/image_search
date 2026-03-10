import styles from './TabNav.module.css'

interface TabNavProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

const TABS = [
  { id: 'search', label: 'Search' },
  { id: 'ingest', label: 'Ingest' },
]

export default function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <nav className={styles.nav}>
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  )
}
