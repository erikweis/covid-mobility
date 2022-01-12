# Introduction

The crisis presented by COVID-19, and the sweeping public health measures required to slow its spread, have necessitated widespread changes to human behavior across the United States. For example, the COVID-19 pandemic has seemingly induced a widespread reshuffling of where people choose to live \citep{wu2021americans}. This reshuffling has been driven largely by the dramatic shift in lifestyle due to social distancing measures and remote work. Amidst this upheaval, housing prices also soared, to varying degrees, throughout the United States \cite{anenberg2021housing}. 

We want to understand the nature of this dramatic shift. In particular, how does an increased opportunity for remote work and increased desire for suburban and rural housing accommodation impact the demographics of communities in the United States? Our research team implemented an agent-based model of household mobility patterns prior and subsequent to the rise of the COVID-19 pandemic in an effort to simulate the impact on both intercommunity and household-level mobility dynamics, as well as community-level socioeconomic heterogeneity, of COVID-motivated changes to household income, preferences for low population-density communities, and the ability to work from home. Our model simulates the key mechanisms of these mobility shifts and explores the long-term impacts of the move on the demographics of communities.

# Empirical Motivations

Changes to personal mobility patterns rising out of the pandemic have been extensively documented,\cite{chang2021mobility} and agent-based modeling of housing mobility is not new. \cite{jordan2012agent} \cite{gulden2011modeling} Our work differs from prior approaches in two key respects: (1) while most work on mobility focuses on movement within specific cities, we take a more generalized and macroscopic view, incorporating moves between idealized cities, rural, and suburban areas, and  (2) we consider the impact of COVID-19 specifically on mobility dynamics. Our model was developed to simulate some of the drivers of household mobility identified by the research, which has shown that household moves in the United States are governed by two principal causes: 1) job opportunity and 2) improved living accommodations (taking into account factors such as cost, location, and quality).  It also incorporates other real-world behavioral phenomena, such as the relative frequency and distance of household moves as distributed across varying income levels. 

# Model Summary

## Overview \& Process
Our household mobility model  simulates the movement of agents, or households, within a finite geographic grid, or environment, composed of heterogeneous locations (e.g., cities, suburbs, towns). The primary components of the simulation are the simulation itself (consisting of the environment and the functions governing its interactions with agents and the data it generates), the agents, and the locations. Agents and locations are characterized by multiple variable and invariant attributes, and simulations by multiple tunable parameters that affect the operation of the methods defined for agents and locations according to specific environmentally-defined conditions. 

Our early efforts at model development created a functional structure consisting of the environment and simplified versions of locations and agents; we approximated real-world initialization distributions and piloted mechanisms based on straightforward linear combinations of variables representing move probabilities and location selection processes. As our model demonstrated a baseline capacity to function as intended and to generate interesting behavior, we began to integrate nonlinear, interdependent equations in an effort to represent more complex dynamics among model components and to embed within the structure richer assumptions about the conditions governing agents' decision-making processes.  

## Modeling Assumptions
We embed certain assumptions into our model, specifically: (1) The location being modeled has periodic boundaries,  (2) The total number of locations and agents remains constant, representing a closed system, (3) Certain attributes for each component are fixed for the duration of the simulation: for example, a location’s quality score and population capacity, (4) job opportunity can be approximated through a location’s population (i.e., more populated areas will have greater job opportunity, and (5) The COVID-19 pandemic presented system-wide changes to certain agent preferences (e.g., population density of home location), attributes (household income), and behaviors (increased radius for job-related moves).

### Structure
We consider a model of $N$ agents that occupy sites, called locations, on regular grid of size $g \times g$. We refer to the set of all agents as $A = \{ A_i \}$ and all locations as $L = \{L_{xy}\}$. Agents move from location to location according to stochastic processes defined below.

## Initial State

### Locations

To begin our simulation, we initialize each location $L_{xy}$ with a capacity, $s_{xy}$, which represents the maximum number of agents who can occupy location $(x,y)$. These capacities remain invariant throughout the simulation. To establish the landscape, we place $2g$ "cities" $\{V_i\}$ on the grid, where each city $V_i$ decays as a multivariate normal centered about a randomly chosen location $x,y$ and covariance matrix $\text{diag}(w_i,w_i)$. The size of the city $k_i$ is drawn from an inverse powerlaw distribution with exponent $2.5$, and its width is $w_i = \log(k_i)/2$. The total capacity at each location $(x,y)$ is given by
$$s_{xy} \propto \sum_{i=1}^{2g} k_i V_i(x,y).$$
The capacities $\{ s_{xy} \}$ are normalized such that
$$s_{total} = \sum_{xy} s_{xy} = \phi_{\text{global}} N,$$
where $\phi_{\text{global}}$ is the global occupancy of the system. Figure \ref{location_capacities} shows an example of the initial state. 

% [INSERT INITIALIZEDINCOMEDISTRIBUTION.PNG]
\begin{figure}[H]
    \centering
    \includegraphics[width=8cm]{figures/capacities.png}
    \caption{Sample of initialized capacity distribution for locations.}
    \label{location_capacities}
\end{figure}

Each location $j = (x,y)$ has the following properties:
\begin{itemize}
    \item A median income $I^{*}_j$ at any point, represents the median income of all agents currently living at location $j$.
    \item The occupancy of the location $\phi_j \in [0,1]$ represents the ratio of the number of agents living at location $j$ to the locations capacity $s_j$.
    \item The cost of housing $h$ is given by
    $$h_j = 0.3 I^{*}_j + \gamma \frac{\phi}{1-\phi}$$. 
\end{itemize}
The above function for housing cost assumes that prices depend on what occupants are willing to pay, and we assume that median income is an important factor in determining that maximum price. We set housing cost as  30\% of the median income of a location's population, a rate that is defined by the United States Department of Housing and Urban Development to be 'affordable' \cite{hud2006hud}. The housing cost function also depends on occupancy, or the ratio of a location’s point-in-time population to its initialized capacity, which we consider to be a proxy for demand. Hence, as $\phi \to 1$, we expect housing cost to increase steeply, and for the purposes of our model, $h_j \to \infty$ in this limit.

For the purposes of the simulation, all prices are relative, and so while the exact value of housing cost is not paramount, the parameter $\gamma$ does play an important role in determining the relative importance of fluctuating demand versus demographic makeup of a location. We choose $\gamma = 1500$.

### Agents

We initialize each agent $A_i$ with the following properties:
\begin{itemize}
    \item The agent's income $I_i$ is drawn from an inverse power law $p(I) = a I^{-a}$. The agent's income remains fixed throughout the simulation.
    \item Preferred population size $s'_i$, is drawn randomly from a distribution $p(s') = \text{LogNormal}(\bar{s},0.5)$
    \item A remote work status $q$, is equal to $1$ if working remotely and $0$ otherwise.
\end{itemize}
As each agent's income and preferred population size are drawn independently, we assume no correlation of these variables.

## Model Dynamics

At each time step $t$, agent $i$ is updated according to the following process:
\begin{enumerate}
    \item With probability $P(L,A)$, agent $i$ decides to move. If the agent does not move, their update is complete.
    \item If moving, the agent selects $n$ search locations using a locally-biased random search of their surroundings.
    \item The agent chooses the optimal move location according to a cost function $C$ which encodes their preferences.
    \item The agent then moves to the optimal location.
\end{enumerate}

## Deciding whether to move
We calculate the probability of moving with a second cost function, which is given by
$$D_i = \alpha \frac{h_{L_i}}{I_i}.$$
We then calculate the probability using a sigmoid function
$$p(D_i) = \frac{1}{1+e^{-(m D_i - b)}},$$

where $m=1.25$ and $b=6$. These values were tuned according to the model output such that the overall move rate is reasonably low so that moves to simultaneous locations are unlikely and updates can be done asynchronously.

## Choosing a new location

Once agents decide to move, they select $n$ locations to search, where $n$ is a function that depends on income:
$$n(I) = \text{int}(3 + m_n I),$$

and $m=0.00011$ is a coefficient ensuring that an income of $\$200,000$ has $25$ location choices. This choice captures an assumption that higher income people are more capable of exerting time and resources on finding an optimal place to live.

The locations are selected by choosing an angle $\theta ~ \text{Uniform}(0,2\pi)$ and a distance $d ~ \text{Power}(\xi)$, where $xi$ also depends on income as
$$\xi(I) = 1.2 + m_{\xi} I$$
and $m_{\xi} = 0.000019$ is set such that the $\xi(200,000) = 4$. Together, $d$ and $\theta$ define a location relative to the agent. These values are discretized and the closest grid space is found. The dependency of $\xi$ on income also captures an assumption that higher income people have the resources to move farther than lower-income individuals. Figure \ref{move_choice} shows examples of how move choices might look for various agent incomes.

\begin{figure*}
    \centering
    \includegraphics[width=0.9\textwidth]{figures/move_possibilities_by_income.png}
    \caption{Examples of move choices broken down by agent income.}
    \label{move_choice}
\end{figure*}

From the $n(I_i)$ search locations, each agent chooses the location that maximizes the cost function $C_i(L_j)$. This function summarizes all the preferences of an agent, and depends on the following factors:
\begin{itemize}
    \item The fit between the agent's preferred population density and that of the location in question
    \item A global preference for higher location capacities, which we describe below as a proxy for the greater job opportunity in higher-density areas, such as cities. Moreover, this effect should not occur for remote workers, who can work from anywhere.
    \item A preference for living in a location where the median income is as close to the agent's income. We can interpret this as a homophily parameter that indirectly encourages clustering of agents of similar income to live together. We interpret this mechanism as capturing the desire of agents to live within their means, not below or above.
    \item A preference for lower housing cost. This term is down-weighted by income, such that higher-income agents are less influenced by this effect.
\end{itemize}
These factors are quantified mathematically as 
\begin{multline*}
    C_i(L_j) = \beta_0 \left|s' - s_j\right| + \beta_1 (1-q) s_j + \\ \beta_2 |I^{*}_j - I_i| - \beta_3 \frac{h}{I_i}.
\end{multline*}
The values of $\beta_i$ are chosen in a somewhat ad hoc fashion such that the typical values of each term of the cost function are roughly equivalent. However, deviation from these baseline values increases the relative impact of the term. For example, increasing $\beta_1$ would increase agent's preference to live in a city.

Importantly, an agent's current location is always included in the list of search locations, so at each time step, agent's might want to move, but choose not to because their search did not reveal a more optimal living location.

We chose these mechanisms as a minimal description of the model mechanisms we wished to consider. In particular, we are interested in the tension between desire to live with proximity to a city but also with a lower population density. This tension between the job opportunity of cities and affordable or lower-density living accommodation is explicitly a trade-off that agents negotiate.

## Introducing COVID-19
We introduce COVID-19 as an exogenous shock to the housing system, whereby agents' preferences change suddenly and all at once. At time $t^*$, we set all agents with income greater than a threshold $I_{rt}$ as remote. This alleviates the aforementioned tension between living in cities for job opportunity and a desire to live in lower-population areas.

Because we were particularly curious about the influx of city dwellers to suburban and rural areas due to the rise of remote work, we did not make any other adjustments to the model, though many possibilities might make sense. For instance, one could justify adjusting making an instantaneous adjustment of every agent's preferred population density to more rural areas, since the appeal of living in cities waned with the closing of restaurants, museums, and all the other attractions of a city.

# Simulations

We run simulations with the following parameters: $N=10,000$, $g=20$, $t_{max} = 1000$, $t^* = 500$, $I_{rt} = 0$. We also simulated the model with both a power law and exponential income distribution.

Our code is available at https://github.com/erikweis/covid-mobility.

# Results

## Population re-distribution due to COVID-19

The most significant finding of our model is the phenomenon of pandemic-precipitated suburban flight, which begins immediately upon introduction of COVID-19 and reverses many of the pre-pandemic mobility trends. Animated heatmaps of our simulation environment over time reveal that agents across all income quintiles tend to congregate in high capacity locations. To gauge demand for locations before and after COVID, we can look at occupancy (Figure \ref{occupancy_prepost}, \ref{fig:pop_dens_occupancy}).

\begin{figure}
    \centering
    \includegraphics[width=0.99\linewidth]{figures/OccupancyPrePostCOVID.png}
    \caption{The change in mean occupancy before and after the introduction of COVID-19. The occupancy of high capacity locations decreases, while occupancy of lower capacity locations generally increases.}
    \label{occupancy_prepost}
\end{figure}

The introduction of COVID in our model indirectly emphasizes agents’ preferences for population density by explicitly eliminating a preference for high capacity locations, households begin to redistribute themselves away from high capacity locations (large urban centers) to middle-density locations (similar to suburbs).

As shown in Figure \ref{fig:spatial_distribution_by_income}, the distribution of the population depends significantly on income. Before COVID, agents are all concentrated in higher capacity locations, but the lowest income bracket cannot live in the low capacity locations, presumably because the cost of housing is too high. This behavior persists in the COVID era, despite how (somewhat unrealistically) low-income agents also working remotely.

\begin{figure*}
    \centering
    \includegraphics[width=0.9\linewidth]{figures/prepostCOVIDincome.png}
    \caption{Distributions of agents by income at t $\approx$ 0, 499, 750, 999}
    \label{fig:spatial_distribution_by_income}
\end{figure*}

\begin{figure}
    \centering
    \includegraphics[width=0.9\linewidth]{figures/pop_dens_post.png}
    \includegraphics[width=0.9\linewidth]{figures/pop_dens_pre.png}
    \caption{Population density and occupancy before (above) and after (below) the introduction of COVID-19. We see a dispersion from city centers towards lower size locations.}
    \label{fig:pop_dens_occupancy}
\end{figure}

## Move Activity

Our model begins with a flurry of move activity as the model reaches an apparent equilibrium. This is because agents are randomly assigned locations, regardless of their income. Once this initial activity quells, we see roughly stable behavior until the introduction of COVID-19, where a bunch of activity occurs before reaching a new equilibrium (Figure \ref{fig:move_activity}).

\begin{figure}
    \centering
    \includegraphics[width=0.95\linewidth]{figures/move_activity_by_income_bracket.png}
    \caption{Move activity by income quintile. The introduction of COVID-19 at time step 500 is clearly visible.}
    \label{fig:move_activity}
\end{figure}


## Distribution of Agent Incomes

The distribution of agent incomes matters a great deal in determining clustering of similar-income agents in the same location. We originally started with an exponential distribution, but modified our model to a power law distribution.With this change, we observed an overall increase in the distribution of median incomes, as shown in Figure \ref{fig:changes_median_income}. We assume that the small proportion of high income agents with an exponential income distribution was not significant enough to price out lower income agents from the most desired locations.

\begin{figure}[h]
    \centering
    \includegraphics[width=8cm]{figures/ChangesToMedianIncome.png}
    \caption{The distribution of median incomes, both before and after the introduction of COVID-19 is narrower for an exponential income distribution (bottom) compared to a power-law distribution (top).}
    \label{fig:changes_median_income}
\end{figure}

## Housing Cost

The model's calculation of housing cost depends on two separate factors: the median income of a location and the occupancy. To look at the pressure of housing price on model dynamics, we look at the housing score portion to the overall contribution and observe that COVID initially sees a relaxation of prices. However, this is likely due to higher income agents moving to lower-income areas. After a reshuffling period, we see higher housing cost pressure than at equilibrium before COVID. These dynamics are shown in Figure \ref{fig:housing_cost}.

\begin{figure}[h]
    \centering
    \includegraphics[width=7cm]{figures/ChangesToHousingCost.png}
    \caption{The cost function dictating agent's probability of moving is driven by a term that depends on housing cost and agent income. That term is shown here in orange. Because our final model only depends on housing cost, this is also equivalent to the total score.}
    \label{fig:housing_cost}
\end{figure}

# Discussion and future work

With agent based models, the number of assumptions expands considerably and quickly. Moreover, balancing a correct representation of the underlying mechanisms with model simplicity is eminently challenging. We assume a distribution of preferred location sizes. This might be empirically motivated, but ultimately is just a starting point. This distribution certainly impacted the model considerably. We'd need to do more analysis to understand the impact of this preference distribution. Does the suburban flight we observed depend exclusively on how this preference distribution is set, or does the phenomena occur for a wide range of preference distributions. Despite this potential case of assumptions directly prescribing the outcomes of the model, we feel that this preference distribution is well-motivated with empirical observations. 

Even if we directly caused the suburban flight by carefully setting initial conditions, our model has value in that we can now explore alternative scenarios for remote work. In the future, we'd like to consider what happens if we start to re-open the simulation (by once again removing remote work). Would the state of the model recover it's pre-COVID arrangement?

Our choice of mechanisms evolved as we added to our model. One early mistake we made was that initially, our model of housing price did not go infinity as occupancy reached and exceeded one. Moreover, agents did not consider the possible effects of housing cost until actually deciding to move to a location. This lead to a few locations being extremely desirable. These locations had populations many times their capacity limit, and agents moved back and forth between an undesirable location, where housing cost was cheaper, and a desirable location, which was too full to accommodate anyone. We solved this problem in two ways. First we implemented a fixed cutoff that no agents could move to a location with occupancy greater than one. Secondly, we allowed agents who decided to move to choose to remain in their current location, if it was better than any of the search locations. 

While we did some preliminary measures of spatial income inequality, we were not confident in our metrics. We expect this will be a key summary statistic for evaluating the state of the model before and after COVID. As our model assumes a fixed income distribution, traditional scores such as the Gini coefficient are insufficient to answer the questions of interest.

There are several kernels of interest that merit further exploration. Firstly, what is it about the post-COVID configuration that leads to a steady-state but with higher pressure on housing cost? We'd also like to do some parameter sweeps on various model parameters. For instance, what happens if we emphasize or de-emphasize some of terms in the location scores. For example, how would inequality increase or decrease if the "clustering" term were much stronger. We'd also like to explore the role of global occupancy on the model. If there are is housing availability in the system overall, do the dynamics of the model change? Finally, we only every looked at the scenario when all agents have the opportunity to work remotely. We would like to explore what happens when only some agents work remotely, and in particular if these agents are above a certain income threshold.

\begin{figure}
    \centering
    \includegraphics[width=7cm]{figures/ChangeIllustration.png}
    \caption{Life cycle of our ecosystem.}
    \label{change_illustration}
\end{figure}

# Conclusions

We designed our model to capture the impact of remote work on people's choice of living locations. Our model captured the essential observations we observed empirically: flight from city centers to suburbs in search of lower density areas. Changes to our agents’ effective population density preferences and changes to housing cost evolve as the flow of agents between locations were the two most important contributing factors to suburban flight. In turn, agents responded to these environmental updates. This process key dynamic is described in Figure \ref{change_illustration}. While our model adequately produces some behavior, we think more can be done to explore the effects of the model under various scenarios.


% # References}

% HUD Archives: Glossay of Terms to Affordable Housing. (2006) https://archives.hud.gov/local/nv/goodstories/2006-04-06glos.cfm

\bibliographystyle{apsrev4-2}
% \bibliographystyle{abbrvnat}
\bibliography{references}
\end{document}