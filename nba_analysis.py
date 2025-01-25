import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import streamlit as st

# Load datasets
def load_data():
    games = pd.read_csv("games.csv")
    teams = pd.read_csv("teams.csv")
    players = pd.read_csv("players.csv")
    games_details = pd.read_csv("games_details.csv", low_memory=False)
    ranking = pd.read_csv("ranking.csv")
    return games, teams, players, games_details, ranking

# Task 1: Compare Team Performance Across Seasons and Games with enhanced filters
def team_performance_across_seasons(ranking, teams):
    # Map TEAM_ID to TEAM_NAME and SEASON_YEAR
    ranking['TEAM_NAME'] = ranking['TEAM_ID'].map(dict(zip(teams['TEAM_ID'], teams['NICKNAME'])))
    ranking['SEASON_YEAR'] = ranking['SEASON_ID'].astype(str).str[1:].astype(int)

    # Group by SEASON_YEAR and TEAM_NAME to calculate mean win percentage
    team_season_win_pct = ranking.groupby(['SEASON_YEAR', 'TEAM_NAME'])['W_PCT'].mean().reset_index()

    st.subheader("Team Win Percentage Across Seasons")

    # Filters (moved to main area)
    moving_avg_window = st.slider("Moving Average Window (Seasons)", 1, 5, 3)
    team_season_win_pct['W_PCT_SMOOTH'] = team_season_win_pct.groupby('TEAM_NAME')['W_PCT'].transform(
        lambda x: x.rolling(window=moving_avg_window, min_periods=1).mean()
    )

    all_teams = sorted(team_season_win_pct['TEAM_NAME'].unique())
    select_all = st.checkbox("Select All Teams", value=True)

    if select_all:
        selected_teams = all_teams
    else:
        selected_teams = st.multiselect("Select teams to display", all_teams, default=[all_teams[0]])

    # Filter by selected teams
    filtered_data = team_season_win_pct[team_season_win_pct['TEAM_NAME'].isin(selected_teams)]

    # Visualization
    fig_line = px.line(
        filtered_data,
        x='SEASON_YEAR',
        y='W_PCT_SMOOTH',
        color='TEAM_NAME',
        markers=True,
        title="Team Win Percentage (Smoothed)",
        labels={"SEASON_YEAR": "Season Year", "W_PCT_SMOOTH": "Smoothed Win Percentage", "TEAM_NAME": "Team"},
        template="plotly_white"
    )
    fig_line.update_traces(line=dict(width=2), marker=dict(size=6))
    fig_line.update_layout(hovermode="x unified", legend_title="Teams")
    st.plotly_chart(fig_line)

# Task 2: Identify Top-Performing Players
def top_performing_players(games_details, teams):
    st.subheader("🌟 Top 10 Players: Average Points, Rebounds, and Assists")

    # Map TEAM_ID to TEAM_NAME
    games_details['TEAM_NAME'] = games_details['TEAM_ID'].map(dict(zip(teams['TEAM_ID'], teams['NICKNAME'])))

    # Filters (moved to main area)
    if 'POSITION' in games_details.columns:
        positions = sorted(games_details['POSITION'].dropna().unique())
        selected_positions = st.multiselect("Select Player Positions", positions, default=positions)
        games_details = games_details[games_details['POSITION'].isin(selected_positions)]

    if 'SEASON_ID' in games_details.columns:
        all_seasons = sorted(games_details['SEASON_ID'].unique())
        selected_season = st.selectbox("Select a Season", all_seasons)
        games_details = games_details[games_details['SEASON_ID'] == selected_season]

    min_games = st.slider("Minimum Games Played", 0, 82, 10)
    games_details = games_details.groupby('PLAYER_NAME').filter(lambda x: len(x) >= min_games)

    all_teams = sorted(games_details['TEAM_NAME'].unique())
    select_all = st.checkbox("Select All Teams", value=True)

    if select_all:
        selected_teams = all_teams
    else:
        selected_teams = st.multiselect("Select a team to analyze players", all_teams, default=[all_teams[0]])

    if selected_teams:
        filtered_data = games_details[games_details['TEAM_NAME'].isin(selected_teams)]
    else:
        st.warning("Please select at least one team to display the analysis.")
        return

    # Top players analysis
    metrics = ['PTS', 'REB', 'AST']
    sort_by = st.multiselect("Sort By Metrics", options=["Points", "Rebounds", "Assists"], default=["Points"])
    sort_metrics_map = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST"}
    sort_metrics = [sort_metrics_map[metric] for metric in sort_by]

    top_players = (
        filtered_data.groupby(['PLAYER_NAME', 'TEAM_NAME'])[metrics]
        .mean()
        .sort_values(by=sort_metrics, ascending=False)
        .head(10)
        .reset_index()
    )

    top_players['PLAYER_LABEL'] = top_players['PLAYER_NAME'] + " (" + top_players['TEAM_NAME'] + ")"

    bar_width = 0.25
    x = range(len(top_players))

    # Bar plot for top players
    fig_bar, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x, top_players['PTS'], width=bar_width, label='Points', color='royalblue')
    ax.bar([i + bar_width for i in x], top_players['REB'], width=bar_width, label='Rebounds', color='lightgreen')
    ax.bar([i + 2 * bar_width for i in x], top_players['AST'], width=bar_width, label='Assists', color='orange')

    ax.set_xticks([i + bar_width for i in x])
    ax.set_xticklabels(top_players['PLAYER_LABEL'], rotation=45, ha='right', fontsize=12)
    ax.set_title("Top 10 Players: Average Points, Rebounds, and Assists", fontsize=16)
    ax.set_xlabel("Players (Team)", fontsize=14)
    ax.set_ylabel("Average Metrics", fontsize=14)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    st.pyplot(fig_bar)

# Task 3: Home Court Advantage:
def home_court_advantage(games):
    st.subheader("Home vs Away Win Percentage")

    team_id_to_name = {
        1610612737: 'Hawks',
        1610612738: 'Celtics',
        1610612739: 'Cavaliers',
        1610612740: 'Pelicans',
        1610612741: 'Bulls',
        1610612742: 'Mavericks',
        1610612743: 'Nuggets',
        1610612744: 'Warriors',
        1610612745: 'Rockets',
        1610612746: 'Clippers',
        1610612747: 'Lakers',
        1610612748: 'Heat',
        1610612749: 'Bucks',
        1610612750: 'Timberwolves',
        1610612751: 'Nets',
        1610612752: 'Knicks',
        1610612753: 'Magic',
        1610612754: 'Pacers',
        1610612755: '76ers',
        1610612756: 'Suns',
        1610612757: 'Trail Blazers',
        1610612758: 'Kings',
        1610612759: 'Spurs',
        1610612760: 'Thunder',
        1610612761: 'Raptors',
        1610612762: 'Jazz',
        1610612763: 'Grizzlies',
        1610612764: 'Wizards'
    }

    games['HOME_TEAM_NAME'] = games['HOME_TEAM_ID'].map(team_id_to_name)
    games['VISITOR_TEAM_NAME'] = games['VISITOR_TEAM_ID'].map(team_id_to_name)

    # Filters (moved to main area)
    all_teams = sorted(games['HOME_TEAM_NAME'].dropna().unique())
    selected_team = st.selectbox("Choose a team to analyze:", options=["All Teams"] + all_teams, index=0)

    all_seasons = sorted(games['SEASON'].unique())
    selected_seasons = st.slider("Season Range", min(all_seasons), max(all_seasons),
                                 (min(all_seasons), max(all_seasons)))

    filtered_games = games[(games['SEASON'] >= selected_seasons[0]) & (games['SEASON'] <= selected_seasons[1])]

    if selected_team != "All Teams":
        filtered_games = filtered_games[filtered_games['HOME_TEAM_NAME'] == selected_team]

    if not filtered_games.empty:
        home_win_percentage = (filtered_games['HOME_TEAM_WINS'].value_counts(normalize=True) * 100).get(1, 0)
        away_win_percentage = (filtered_games['HOME_TEAM_WINS'].value_counts(normalize=True) * 100).get(0, 0)

        fig_bar, ax = plt.subplots(figsize=(8, 6))
        ax.bar([0, 1], [away_win_percentage, home_win_percentage], color=['orange', 'purple'])
        ax.set_title(f"Home vs Away Win Percentage for {selected_team}")
        ax.set_xlabel("Win Location")
        ax.set_ylabel("Win Percentage")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Away Wins', 'Home Wins'])

        for i, v in enumerate([away_win_percentage, home_win_percentage]):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=12)

        st.pyplot(fig_bar)
    else:
        st.warning("No games match the selected filters.")

# Task 4: Historic Data and Most Recommended Games to Watch
def historic_data_and_recommended_games(games, teams, games_details):
    st.subheader("Historic Data and Most Recommended Games")

    # Ensure necessary columns in teams DataFrame
    if 'CITY' in teams.columns and 'NICKNAME' in teams.columns:
        teams['TEAM_NAME'] = teams['CITY'] + " " + teams['NICKNAME']
    else:
        st.error("Missing 'CITY' or 'NICKNAME' columns in the teams dataset.")
        return

    # Add calculated columns for Total Points and Point Difference
    games['Total Points'] = games['PTS_home'] + games['PTS_away']
    games['Point Difference'] = abs(games['PTS_home'] - games['PTS_away'])

    # Select top 10 most exciting games based on Total Points and Point Difference
    best_games = games.sort_values(
        by=['Total Points', 'Point Difference'], ascending=[False, True]
    ).head(10)

    # Map team names for better visualization
    best_games = best_games.merge(
        teams.rename(columns={'TEAM_ID': 'HOME_TEAM_ID', 'TEAM_NAME': 'Home Team'})[['HOME_TEAM_ID', 'Home Team']],
        on='HOME_TEAM_ID', how='left'
    ).merge(
        teams.rename(columns={'TEAM_ID': 'VISITOR_TEAM_ID', 'TEAM_NAME': 'Visitor Team'})[['VISITOR_TEAM_ID', 'Visitor Team']],
        on='VISITOR_TEAM_ID', how='left'
    )

    fig_games = px.scatter(
        best_games,
        x='Point Difference',
        y='Total Points',
        size='Total Points',
        color='Total Points',
        hover_name='Home Team',
        hover_data={
            'Visitor Team': True,
            'PTS_home': ':.0f',
            'PTS_away': ':.0f',
            'Total Points': ':.0f',
            'Point Difference': ':.0f',
            'SEASON': True
        },
        title="Most Exciting Games in NBA History",
        labels={
            'Point Difference': 'Point Difference (Close Games)',
            'Total Points': 'Total Points Scored',
        },
        color_continuous_scale=px.colors.sequential.Sunset,
        size_max=50
    )

    st.plotly_chart(fig_games)

# Task 5: Discover Correlations Between Team Stats and Wins
def team_stats_correlation(games):
    st.subheader("Correlation Between Team Stats and Wins")

    team_stats = games[[
        'PTS_home', 'REB_home', 'AST_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'HOME_TEAM_WINS'
    ]]
    team_stats.rename(columns={
        'PTS_home': 'Points',
        'REB_home': 'Rebounds',
        'AST_home': 'Assists',
        'FG_PCT_home': 'Field Goal %',
        'FT_PCT_home': 'Free Throw %',
        'FG3_PCT_home': '3-Point %',
        'HOME_TEAM_WINS': 'Wins'
    }, inplace=True)

    correlation_matrix = team_stats.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Between Team Stats and Wins")
    st.pyplot(fig)

# Main function to execute tasks
def main():
    games, teams, players, games_details, ranking = load_data()
    st.set_page_config(page_title="NBA Deep Dive Dashboard", page_icon="🏀")

    st.sidebar.title("Navigation")
    task = st.sidebar.radio("Select Task", [
        "Introduction",
        "Team Performance Across Seasons",
        "Top-Performing Players",
        "Home-Court Advantage",
        "Historic Data and Recommended Games",
        "Team Stats Correlation"
    ])

    # Task 1: Introduction
    if task == "Introduction":
        st.title("🏀 **NBA Deep Dive Dashboard**: The Winning Formula Revealed 🏆")
        st.markdown("""
            ## 🎯 **Unlocking the Secrets of NBA Success**
            Welcome to the ultimate basketball analytics experience! This dashboard takes you courtside into the heart of the NBA, helping you uncover:
            - 🏀 **Seasonal Dynamics**: Which teams stay consistent, and who surprises us year after year?
            - 🌟 **Player Power**: Is it all about the points? Or do rebounds and assists change the game?
            - 🏠 **Home Advantage**: Myth or reality? Does playing on your turf truly matter?
            - 🎯 **Shooting Precision**: Do field goals, free throws, and three-pointers make or break teams?

            ## 📊 **The Data Game Plan**
            Armed with a comprehensive dataset, we’re ready to crunch the numbers:
            - **Games**: Dive into 26,651 matchups with every score, percentage, and outcome.
            - **Rankings**: Explore 210,342 rows of team standings and win-loss stats.
            - **Game Details**: Get granular with player stats across 668,628 rows.
            - **Players**: Analyze insights from 7,228 NBA legends and rising stars.
            - **Teams**: Understand the story of all 30 teams, from cities to names.

            Get ready to explore trends, find the X-factor, and become an NBA analytics pro. Let’s break down the game like never before! 🎉
            """)

    # Task 2: Team Performance Across Seasons
    elif task == "Team Performance Across Seasons":
        st.title("📈 Team Performance Across Seasons")
        st.markdown("""
            ### Overview
            Discover how NBA teams perform across different seasons and conferences. Identify trends, dominant teams, 
            and how win percentages evolve over time. Use filters to analyze specific teams and their season-by-season grind.
            """)
        team_performance_across_seasons(ranking, teams)

    # Task 3: Top-Performing Players
    elif task == "Top-Performing Players":
        st.title("🌟 Top-Performing Players")
        st.markdown("""
            ### Overview
            Dive into individual player performances to identify the top contributors in points, rebounds, and assists.
            Find out if scoring alone guarantees success or if balanced performances across metrics make the difference.
            """)
        top_performing_players(games_details, teams)

    # Task 4: Home-Court Advantage
    elif task == "Home-Court Advantage":
        st.title("🏠 Home-Court Advantage")
        st.markdown("""
            ### Overview
            Is home-court advantage a myth or reality? Explore data to determine whether playing at home significantly 
            improves a team's chances of winning. Compare win percentages for home and away games.
            """)
        home_court_advantage(games)

    # Task 5: Historic Data and Recommended Games
    elif task == "Historic Data and Recommended Games":
        st.title("📜 Historic Data and Recommended Games")
        st.markdown("""
            ### Overview
            Analyze the top-performing teams in each season and uncover the most exciting games to watch. 
            Discover the best teams, standout players, and high-scoring games that defined the NBA's history.
            """)
        historic_data_and_recommended_games(games, teams, games_details)

    # Task 6: Team Stats Correlation
    elif task == "Team Stats Correlation":
        st.title("🔗 Team Stats Correlation")
        st.markdown("""
            ### Overview
            Uncover the relationships between key team stats like points, rebounds, assists, and shooting percentages.
            Learn which metrics are most strongly correlated with winning games and dominating the league.
            """)
        team_stats_correlation(games)

if __name__ == "__main__":
    main()
