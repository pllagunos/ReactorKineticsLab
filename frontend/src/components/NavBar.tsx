import { NavLink } from 'react-router-dom'

export function NavBar() {
  return (
    <nav className="nav-bar" aria-label="Main navigation">
      <div className="nav-bar__inner">
        <span className="nav-bar__brand">⚛ Reactor Sim</span>
        <div className="nav-bar__links">
          <NavLink
            to="/overview"
            className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
          >
            Overview
          </NavLink>
          <NavLink
            to="/core"
            className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
          >
            Core
          </NavLink>
          <NavLink
            to="/multigroup"
            className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
          >
            Multigroup
          </NavLink>
          <NavLink
            to="/transient"
            className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
          >
            Transient
          </NavLink>
          <NavLink
            to="/thermal-hydraulics"
            className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
          >
            Thermal Hydraulics
          </NavLink>
        </div>
      </div>
    </nav>
  )
}
