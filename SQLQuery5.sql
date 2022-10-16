SELECT *
FROM PortafolioProject..CovidDeaths$
WHERE continent IS NOT NULL
ORDER BY 3,4

--SELECT *
--FROM PortafolioProject..CovidVaccinations$
--ORDER BY 3,4

-- Select the data that we are going to be using

SELECT location, date, total_cases, new_cases, total_deaths, population
FROM PortafolioProject..CovidDeaths$
ORDER BY 1,2

--Looking at total cases vs total deaths

SELECT location, date, total_cases,total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
FROM PortafolioProject..CovidDeaths$
WHERE location like '%states%'
ORDER BY 1,2

--Looking Total Cases vs Population 
-- Shows what percentage of population got covid
SELECT location, date, population, total_cases, (total_cases/population)*100 as PositivePercentage
FROM PortafolioProject..CovidDeaths$
--WHERE location like '%states%'
ORDER BY 1,2

--Looking at countries with highest infection rate compared to population
SELECT location, population, MAX(total_cases) AS HighestInfectionCount, MAX(total_cases/population)*100 as PercentagePopulationInfected
FROM PortafolioProject..CovidDeaths$
--WHERE location like '%states%' 
GROUP BY location, population
ORDER BY PercentagePopulationInfected DESC

-- LETS BREAK THINGS DOWN BY CONTINENT
SELECT continent, MAX(cast(total_deaths as int)) as TotalDeathCount
FROM PortafolioProject..CovidDeaths$
--WHERE location like '%states%' 
WHERE continent IS NOT NULL
GROUP BY continent
ORDER BY TotalDeathCount DESC

--Showing countries with Highest Death Count per Population
SELECT location, MAX(cast(total_deaths as int)) as TotalDeathCount
FROM PortafolioProject..CovidDeaths$
--WHERE location like '%states%' 
WHERE continent IS NULL
GROUP BY location
ORDER BY TotalDeathCount DESC

--Showing continents with the highest death count per population
SELECT continent, MAX(cast(total_deaths as int)) as TotalDeathCount
FROM PortafolioProject..CovidDeaths$
--WHERE location like '%states%' 
WHERE continent IS NOT NULL
GROUP BY continent
ORDER BY TotalDeathCount DESC


-- GLOBAL NUMBERS
SELECT SUM(new_cases) AS TotalCases, SUM(cast(new_deaths as int)) AS TotalDeaths, SUM(CAST(new_deaths as int))/SUM(new_cases)*100 as DeathPercentage
FROM PortafolioProject..CovidDeaths$
--WHERE location like '%states%'
WHERE continent IS NOT NULL 
--GROUP BY date
ORDER BY 1,2

--Looking at Total Population vs Vaccinations
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(CONVERT(int, vac.new_vaccinations)) OVER (Partition by dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated
-- (RollingPeopleVaccinated/Population)*100
FROM PortafolioProject..CovidDeaths$ dea
JOIN PortafolioProject..CovidVaccinations$ vac
    ON dea.location = vac.location
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL
ORDER BY 2,3

-- USE CTE

WITH PopvsVAC (continent, location, date, population, new_vaccinations, RollingPeopleVaccinated)
AS
(
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
SUM(CONVERT(int, vac.new_vaccinations)) OVER (Partition by dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated
-- (RollingPeopleVaccinated/Population)*100
FROM PortafolioProject..CovidDeaths$ dea
JOIN PortafolioProject..CovidVaccinations$ vac
    ON dea.location = vac.location
	AND dea.date = vac.date
WHERE dea.continent IS NOT NULL
--ORDER BY 2,3
)
SELECT *, (RollingPeopleVaccinated/population)*100
FROM PopvsVAC


-- TEMP TABLE
DROP TABLE IF EXISTS #PercentPopulationVaccinated
CREATE TABLE #PercentPopulationVaccinated
(
continent nvarchar(255),
location nvarchar(255),
date datetime,
population numeric,
new_vaccinations numeric,
RollingPeopleVaccinated numeric
)
INSERT INTO #PercentPopulationVaccinated
SELECT dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CAST(vac.new_vaccinations AS bigint)) OVER (Partition by dea.location ORDER BY dea.location, dea.date) AS RollingPeopleVaccinated
-- (RollingPeopleVaccinated/Population)*100
FROM PortafolioProject..CovidDeaths$ dea
JOIN PortafolioProject..CovidVaccinations$ vac
    ON dea.location = vac.location
	AND dea.date = vac.date
--WHERE dea.continent IS NOT NULL
--ORDER BY 2,3

SELECT *, (RollingPeopleVaccinated/population)*100
FROM #PercentPopulationVaccinated