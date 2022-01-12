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