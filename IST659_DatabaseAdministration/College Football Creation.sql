------RESET DATABASE------
--Select Database
USE CollegeFootball
--Remove all Tables
DROP TABLE IF EXISTS game_stats, game, coach, player, team_season, season, team, conference, division;
--Remove all Procedures
DROP PROCEDURE IF EXISTS addDivision, addConference, addTeam, addSeason, addCoach, fireCoach, addGame, addPlayer, addStats, updateResults, updateChampionships;
--Drop all Views
DROP VIEW IF EXISTS field_goal_totals, sec_bowl_eligible, field_goal_percentage, game_results, national_championship_coaches, rushing_yard_totals
--Drop all Functions
DROP FUNCTION IF EXISTS getFieldGoalPercentage

------CREATE TABLES------
--Create Division Table
CREATE TABLE division(
	division_id INT IDENTITY NOT NULL,
	division_name VARCHAR(20) NOT NULL,
	CONSTRAINT division_PK PRIMARY KEY (division_id));

--Create Conference Table
CREATE TABLE conference(
	conference_id INT IDENTITY NOT NULL,
	conference_name VARCHAR(30) NOT NULL,
	division_id INT NOT NULL,
	CONSTRAINT conference_PK PRIMARY KEY (conference_id),
	CONSTRAINT conference_FK1 FOREIGN KEY (division_id) REFERENCES division(division_id));

--Create Team Table
CREATE TABLE team(
	team_id INT IDENTITY NOT NULL,
	school_name VARCHAR(50) NOT NULL,
	school_nickname VARCHAR(50) NOT NULL,
	team_wins INT,
	team_losses INT,
	team_national_championships INT,
	team_conference_championships INT,
	conference_id INT NOT NULL,
	CONSTRAINT team_PK PRIMARY KEY (team_id),
	CONSTRAINT team_FK1 FOREIGN KEY (conference_id) REFERENCES conference(conference_id));

--Create Season Table
CREATE TABLE season(
	season_id INT IDENTITY NOT NULL,
	season_year CHAR(4) NOT NULL,
	season_wins INT,
	season_losses INT,
	season_conference_championship VARCHAR(3),
	season_national_championship VARCHAR(3),
	season_bowl_game VARCHAR(50),
	CONSTRAINT season_PK PRIMARY KEY (season_id));

--Create Team_Season Table
CREATE TABLE team_season(
	team_season_id INT IDENTITY NOT NULL,
	team_id INT NOT NULL,
	season_id INT NOT NULL,
	CONSTRAINT team_season_PK PRIMARY KEY (team_season_id),
	CONSTRAINT team_season_FK1 FOREIGN KEY (team_id) REFERENCES team(team_id),
	CONSTRAINT team_season_FK2 FOREIGN KEY (season_id) REFERENCES season(season_id));

--Create Coach Table
CREATE TABLE coach(
	coach_id INT IDENTITY NOT NULL,
	coach_first_name VARCHAR(30) NOT NULL,
	coach_last_name VARCHAR(30) NOT NULL,
	coach_wins INT,
	coach_losses INT,
	coach_first_season CHAR(4) NOT NULL,
	coach_last_season CHAR(4),
	coach_status VARCHAR(7) NOT NULL,
	coach_national_championships INT,
	coach_conference_championships INT,
	CONSTRAINT coach_PK PRIMARY KEY (coach_id));

--Create Game Table
CREATE TABLE game(
	game_id INT IDENTITY NOT NULL,
	opponent VARCHAR(50) NOT NULL,
	points_For INT NOT NULL,
	points_against INT NOT NULL,
	game_result VARCHAR(4) NOT NULL,
	game_type VARCHAR(25) NOT NULL,
	coach_id INT NOT NULL,
	season_id INT NOT NULL,
	CONSTRAINT game_PK PRIMARY KEY (game_id),
	CONSTRAINT game_FK1 FOREIGN KEY (coach_id) REFERENCES coach(coach_id),
	CONSTRAINT game_FK2 FOREIGN KEY (season_id) REFERENCES season(season_id));

--Create Player Table
CREATE TABLE player(
	player_id INT IDENTITY NOT NULL,
	player_first_name VARCHAR(30) NOT NULL,
	player_last_name VARCHAR(30) NOT NULL,
	player_position CHAR(2) NOT NULL,
	player_class CHAR(2) NOT NULL,
	CONSTRAINT player_PK PRIMARY KEY (player_id));

--Create Game_Stats Table
CREATE TABLE game_stats(
	game_stats_id INT IDENTITY NOT NULL,
	pass_completions INT,
	pass_attempts INT,
	pass_completion_percentage DECIMAL(12,4),
	pass_yards INT,
	pass_yards_per_attempt DECIMAL(12,4),
	adjusted_pass_yards_per_attempt DECIMAL(12,4),
	pass_touchdowns INT,
	pass_interceptions INT,
	pass_efficiency_rating DECIMAL(12,4),
	rush_attempts INT,
	rush_yards INT,
	rush_yards_per_attempt DECIMAL(12,4),
	rush_touchdowns INT,
	receptions INT,
	rec_yards INT,
	rec_yards_per_reception DECIMAL(12,4),
	rec_touchdowns INT,
	plays_from_scrimmage INT,
	yards_from_scrimmage INT,
	yards_from_scrimmage_per_play DECIMAL(12,4),
	td_from_scrimmage INT,
	solo_tackles INT,
	assisted_tackles INT,
	total_tackles INT,
	tackles_for_loss INT,
	interceptions INT,
	interception_return_yards INT,
	interception_return_yards_per_interception DECIMAL(12,4),
	interception_return_touchdowns INT,
	passes_defended INT,
	fumbles_recovered INT,
	fumble_recovery_return_yards INT,
	fumble_recovery_return_touchdowns INT,
	fumbles_forced INT,
	kickoff_returns INT,
	kickoff_return_yards INT,
	kickoff_return_yards_per_return DECIMAL(12,4),
	kickoff_return_touchdowns INT,
	punt_returns INT,
	punt_return_yards INT,
	punt_return_yards_per_return DECIMAL(12,4),
	punt_return_touchdowns INT,
	extra_points_made INT,
	extra_point_attempts INT,
	extra_point_percentage DECIMAL(12,4),
	field_goals_made INT,
	field_goal_attempts INT,
	field_goal_percentage DECIMAL(12,4),
	kicking_points INT,
	punts INT,
	punting_yards INT,
	punting_yards_per_punt DECIMAL(12,4),
	game_id INT NOT NULL,
	player_id INT NOT NULL,
	CONSTRAINT game_stats_PK PRIMARY KEY (game_stats_id),
	CONSTRAINT game_stats_FK1 FOREIGN KEY (game_id) REFERENCES game(game_id),
	CONSTRAINT game_stats_FK2 FOREIGN KEY (player_id) REFERENCES player(player_id));

------CREATE PROCEDURES------
--Add Division
GO
CREATE PROCEDURE addDivision(@division_name varchar(20))
AS
BEGIN
	--Create Division
	INSERT INTO division(division_name)
	VALUES (@division_name)
END
GO

--Add Conference
CREATE PROCEDURE addConference(@conference_name varchar(30), @division_name varchar(20))
AS
BEGIN
	--Get Division ID
	DECLARE @division_id int
	SELECT @division_id = division_id FROM division
	WHERE @division_name = division_name

	--Create Conference
	INSERT INTO conference(conference_name, division_id)
	VALUES (@conference_name, @division_id)
END
GO

--Add Team
CREATE PROCEDURE addTeam(@school_name varchar(50), @school_nickname varchar(50), @conference_name varchar(30))
AS
BEGIN
	--Get Conference ID
	DECLARE @conference_id int
	SELECT @conference_id = conference_id FROM conference
	WHERE @conference_name =conference_name	

	--Create Team
	INSERT INTO team(school_name, school_nickname, conference_id)
	VALUES (@school_name, @school_nickname, @conference_id)
END
GO

--Add Season
CREATE PROCEDURE addSeason(@school_name varchar(50), @season_year char(4))
AS
BEGIN
	--Create Season
	INSERT INTO season(season_year)
	VALUES (@season_year)

	--Get Season ID
	DECLARE @season_id int 
	SET @season_id = @@IDENTITY

	--Get Team ID
	DECLARE @team_id int
	SELECT @team_id = team_id FROM team
	WHERE school_name = @school_name

	--Create Team_Season
	INSERT INTO team_season(team_id, season_id) 
	VALUES (@team_id, @season_id)
END
GO

--Add Coach
CREATE PROCEDURE addCoach(@coach_first_name varchar(30), @coach_last_name varchar(30), @coach_first_season char(4))
AS
BEGIN
	INSERT INTO coach(coach_first_name, coach_last_name, coach_first_season, coach_status)
	VALUES (@coach_first_name, @coach_last_name, @coach_first_season, 'Current')
END
GO

--Add Game
CREATE PROCEDURE addGame(@team varchar(50), @opponent varchar(50), @year char(4), @game_type varchar(25), @coach_first_name varchar(30), @coach_last_name varchar(30), @points_for int, @points_against int)
AS
BEGIN

	--Get Coach ID
	DECLARE @coach_id int
	SELECT @coach_id = coach_id FROM coach WHERE coach_first_name = @coach_first_name AND coach_last_name = @coach_last_name

	--Get Team ID
	DECLARE @team_id INT
	SELECT @team_id = team_id FROM team WHERE school_name = @team

	--Get Season
		DECLARE @season_id int

		--Get All Team_Season for Team
			--Create temporary table	
			CREATE TABLE one_team(
				one_team_id INT IDENTITY,
				season_id INT, 
				season_year char(4))

			--Add Season ID's to temporary table
			INSERT INTO one_team(season_id)
			SELECT season_id FROM team_season WHERE team_id = (SELECT team_id FROM team WHERE school_name like '%' + @team + '%')

			--Add Season Years to temporary table
			UPDATE one_team 
			SET season_year = season.season_year
			FROM one_team
			JOIN season
			ON season.season_id = one_team.season_id

			--Get Correct Year
			SELECT @season_id = season_id FROM one_team
			WHERE season_year = @year

			--Delete temporary Table
			DROP TABLE one_team

	--Get Result
	DECLARE @game_result varchar(4)
	IF (@points_for > @points_against)
	BEGIN
		SET @game_result = 'Win'
		--Update Season
		Update Season
		SET season_wins = season_wins + 1
		WHERE season_id = @season_id
		--Update Team
		Update Team
		SET team_wins = team_wins + 1
		WHERE team_id = @team_id
		--Update Update Coach
		UPDATE Coach
		SET coach_wins = coach_wins + 1
		WHERE coach_id = @coach_id
		--Update National Championship
		IF @game_type = 'National Championship'
		BEGIN
			Update Season
			SET season_national_championship = season_national_championship + 1
			WHERE season_id = @season_id
			--Update Team
			Update Team
			SET team_national_championships = team_national_championships + 1
			WHERE team_id = @team_id
			--Update Update Coach
			UPDATE Coach
			SET coach_national_championships = coach_national_championships + 1
			WHERE coach_id = @coach_id
		END
		ELSE IF @game_type = 'Conference Championship'
		BEGIN
		--Update Conference Championship
			Update Season
			SET season_conference_championship = season_conference_championship + 1
			WHERE season_id = @season_id
			--Update Team
			Update Team
			SET team_conference_championships = team_conference_championships + 1
			WHERE team_id = @team_id
			--Update Update Coach
			UPDATE Coach
			SET coach_conference_championships = coach_conference_championships + 1
			WHERE coach_id = @coach_id
		END
	END
	ELSE 
	BEGIN
		SET @game_result = 'Loss'
		--Update Season
		Update Season
		SET season_losses = season_losses + 1
		WHERE season_id = @season_id
		--Update Team
		Update Team
		SET team_losses = team_losses + 1
		WHERE team_id = @team_id
		--Update Update Coach
		UPDATE Coach
		SET coach_losses = coach_losses + 1
		WHERE coach_id = @coach_id
	END

	--CREATE Game
	INSERT INTO game(season_id, opponent, points_For, points_against, game_result, game_type, coach_id)
	VALUES (@season_id, @opponent, @points_for, @points_against, @game_result, @game_type, @coach_id) 
END
GO

--Add Player
CREATE PROCEDURE addPlayer(@player_first_name varchar(30), @player_last_name varchar(30), @player_class char(2), @player_position char(2))
AS
BEGIN
	--CREATE Player
	INSERT INTO player(player_first_name, player_last_name, player_position, player_class)
	VALUES(@player_first_name, @player_last_name, @player_class, @player_position)
END
GO

--Add Stats
CREATE PROCEDURE addStats(@player_first_name varchar(30), @player_last_name varchar(30), @team varchar(50), @year CHAR(4), @opponent varchar(50), @pass_completions INT,
	@pass_attempts INT,
	@pass_completion_percentage DECIMAL(12,4),
	@pass_yards INT,
	@pass_yards_per_attempt DECIMAL(12,4),
	@adjusted_pass_yards_per_attempt DECIMAL(12,4),
	@pass_touchdowns INT,
	@pass_interceptions INT,
	@pass_efficiency_rating DECIMAL(12,4),
	@rush_attempts INT,
	@rush_yards INT,
	@rush_yards_per_attempt DECIMAL(12,4),
	@rush_touchdowns INT,
	@receptions INT,
	@rec_yards INT,
	@rec_yards_per_reception DECIMAL(12,4),
	@rec_touchdowns INT,
	@plays_from_scrimmage INT,
	@yards_from_scrimmage INT,
	@yards_from_scrimmage_per_play DECIMAL(12,4),
	@td_from_scrimmage INT,
	@solo_tackles INT,
	@assisted_tackles INT,
	@total_tackles INT,
	@tackles_for_loss INT,
	@interceptions INT,
	@interception_return_yards INT,
	@interception_return_yards_per_interception DECIMAL(12,4),
	@interception_return_touchdowns INT,
	@passes_defended INT,
	@fumbles_recovered INT,
	@fumble_recovery_return_yards INT,
	@fumble_recovery_return_touchdowns INT,
	@fumbles_forced INT,
	@kickoff_returns INT,
	@kickoff_return_yards INT,
	@kickoff_return_yards_per_return DECIMAL(12,4),
	@kickoff_return_touchdowns INT,
	@punt_returns INT,
	@punt_return_yards INT,
	@punt_return_yards_per_return DECIMAL(12,4),
	@punt_return_touchdowns INT,
	@extra_points_made INT,
	@extra_point_attempts INT,
	@extra_point_percentage DECIMAL(12,4),
	@field_goals_made INT,
	@field_goal_attempts INT,
	@field_goal_percentage DECIMAL(12,4),
	@kicking_points INT,
	@punts INT,
	@punting_yards INT,
	@punting_yards_per_punt DECIMAL(12,4))
AS
BEGIN
	--Get Player ID
	Declare @player_id int
	SELECT @player_id = player_id FROM player WHERE player_first_name = @player_first_name AND player_last_name = @player_last_name
	
	--Get Season
		DECLARE @season_id int

		--Get All Team_Season for Team
			--Create temporary table	
			CREATE TABLE one_team(
				one_team_id INT IDENTITY,
				season_id INT, 
				season_year char(4))

			--Add Season ID's to temporary table
			INSERT INTO one_team(season_id)
			SELECT season_id FROM team_season WHERE team_id = (SELECT team_id FROM team WHERE school_name like '%' + @team + '%')

			--Add Season Years to temporary table
			UPDATE one_team 
			SET season_year = season.season_year
			FROM one_team
			JOIN season
			ON season.season_id = one_team.season_id

			--Get Correct Year
			SELECT @season_id = season_id FROM one_team
			WHERE season_year = @year

			--Delete temporary Table
			DROP TABLE one_team


	--Get Game ID
	Declare @game_id int
	SELECT @game_id = game_id FROM game WHERE season_id = @season_id AND opponent = @opponent

	--Create Stats
	INSERT INTO game_stats (pass_completions,
		pass_attempts,
		pass_completion_percentage,
		pass_yards,
		pass_yards_per_attempt,
		adjusted_pass_yards_per_attempt,
		pass_touchdowns,
		pass_interceptions,
		pass_efficiency_rating,
		rush_attempts,
		rush_yards,
		rush_yards_per_attempt,
		rush_touchdowns,
		receptions,
		rec_yards,
		rec_yards_per_reception,
		rec_touchdowns,
		plays_from_scrimmage,
		yards_from_scrimmage,
		yards_from_scrimmage_per_play,
		td_from_scrimmage,
		solo_tackles,
		assisted_tackles,
		total_tackles,
		tackles_for_loss,
		interceptions,
		interception_return_yards,
		interception_return_yards_per_interception,
		interception_return_touchdowns,
		passes_defended,
		fumbles_recovered,
		fumble_recovery_return_yards,
		fumble_recovery_return_touchdowns,
		fumbles_forced,
		kickoff_returns,
		kickoff_return_yards,
		kickoff_return_yards_per_return,
		kickoff_return_touchdowns,
		punt_returns,
		punt_return_yards,
		punt_return_yards_per_return,
		punt_return_touchdowns,
		extra_points_made,
		extra_point_attempts,
		extra_point_percentage,
		field_goals_made,
		field_goal_attempts,
		field_goal_percentage,
		kicking_points,
		punts,
		punting_yards,
		punting_yards_per_punt,
		game_id,
		player_id)
	VALUES (@pass_completions,
		@pass_attempts,
		@pass_completion_percentage,
		@pass_yards,
		@pass_yards_per_attempt,
		@adjusted_pass_yards_per_attempt,
		@pass_touchdowns,
		@pass_interceptions,
		@pass_efficiency_rating,
		@rush_attempts,
		@rush_yards,
		@rush_yards_per_attempt,
		@rush_touchdowns,
		@receptions,
		@rec_yards,
		@rec_yards_per_reception,
		@rec_touchdowns,
		@plays_from_scrimmage,
		@yards_from_scrimmage,
		@yards_from_scrimmage_per_play,
		@td_from_scrimmage,
		@solo_tackles,
		@assisted_tackles,
		@total_tackles,
		@tackles_for_loss,
		@interceptions,
		@interception_return_yards,
		@interception_return_yards_per_interception,
		@interception_return_touchdowns,
		@passes_defended,
		@fumbles_recovered,
		@fumble_recovery_return_yards,
		@fumble_recovery_return_touchdowns,
		@fumbles_forced,
		@kickoff_returns,
		@kickoff_return_yards,
		@kickoff_return_yards_per_return,
		@kickoff_return_touchdowns,
		@punt_returns,
		@punt_return_yards,
		@punt_return_yards_per_return,
		@punt_return_touchdowns,
		@extra_points_made,
		@extra_point_attempts,
		@extra_point_percentage,
		@field_goals_made,
		@field_goal_attempts,
		@field_goal_percentage,
		@kicking_points,
		@punts,
		@punting_yards,
		@punting_yards_per_punt,
		@game_id,
		@player_id)
END
GO

--Fire Coach
CREATE PROCEDURE fireCoach(@coach_first_name varchar(30), @coach_last_name varchar(30), @coach_last_season char(4))
AS
BEGIN
	--Get Coach ID
	DECLARE @coach_id int
	SELECT @coach_id = coach_id FROM coach
	WHERE coach_first_name = @coach_first_name AND coach_last_name = @coach_last_name

	--Set Status to Former
	UPDATE coach SET coach_status = 'Former'
	WHERE coach_id = @coach_id

	--Set Last Season
	UPDATE coach SET coach_last_season = @coach_last_season
	WHERE coach_id = @coach_id
END
GO

------CREATE VIEWS------
--Get Field Goal Totals
CREATE VIEW field_goal_totals
AS SELECT season.season_id, player_first_name, player_last_name, Sum(field_goal_attempts) AS sum_field_goal_attempts, Sum(field_goals_made) AS sum_field_goals_made
FROM ((game_stats INNER JOIN player ON game_stats.player_id = player.player_id) INNER JOIN game ON game_stats.game_id = game.game_id) INNER JOIN ((team_season INNER JOIN season ON team_season.season_id = season.season_id) INNER JOIN team ON team_season.team_id = team.team_id) ON game.season_id = season.season_id
WHERE (((game_stats.field_goal_attempts)>0))
GROUP BY player.player_first_name, player.player_last_name, season.season_id;
GO

--Get Bowl Eligible Teams
CREATE VIEW sec_bowl_eligible
AS SELECT school_name, season_wins, season_losses
FROM (team_season INNER JOIN season ON team_season.season_id = season.season_id) INNER JOIN (team INNER JOIN conference ON team.conference_id = conference.conference_id) ON team_season.team_id = team.team_id
WHERE season_wins >= 6
GO

--Get Best Field Goal Percentage
CREATE VIEW field_goal_percentage
AS 
SELECT player_first_name, player_last_name, CONVERT(DECIMAL(12,4), sum_field_goals_made)/CONVERT(DECIMAL(12,4),sum_field_goal_attempts) AS FieldGoalPercentage
FROM field_goal_totals;
GO

--Get Game Results
CREATE VIEW game_results
AS
SELECT school_name, opponent, game_result, points_For, points_against, season_year
FROM game INNER JOIN ((team_season INNER JOIN season ON team_season.season_id = season.season_id) INNER JOIN team ON team_season.team_id = team.team_id) ON game.season_id = season.season_id
GO

--Get National Championship Coaches
CREATE VIEW national_championship_coaches
AS
SELECT season_year, coach_first_name, coach_last_name, school_name
FROM (team_season INNER JOIN ((game INNER JOIN coach ON game.coach_id = coach.coach_id) INNER JOIN season ON game.season_id = season.season_id) ON team_season.season_id = season.season_id) INNER JOIN team ON team_season.team_id = team.team_id
WHERE game_result='Win' AND game_type='National Championship'
GO

--Get Rushing Yard Totals
CREATE VIEW rushing_yard_totals
AS
SELECT Sum(rush_yards) AS sum_rush_yards, player_first_name, player_last_name, school_name
FROM ((((game_stats INNER JOIN player ON game_stats.player_id = player.player_id) INNER JOIN game ON game_stats.game_id = game.game_id) INNER JOIN season ON game.season_id = season.season_id) INNER JOIN team_season ON season.season_id = team_season.season_id) INNER JOIN team ON team_season.team_id = team.team_id
GROUP BY player.player_first_name, player.player_last_name, team.school_name
GO

------CREATE FUNCTIONS------
--Get Field Goal Percentage for Season
CREATE FUNCTION getFieldGoalPercentage(@player_first_name varchar(30), @player_last_name varchar(30), @team varchar(50), @season_year char(4))
RETURNS DECIMAL(12,4)
AS
BEGIN
	--get season
	DECLARE @season_id INT
	SELECT @season_id = season_id FROM team_season WHERE season_id = (SELECT season_id FROM season WHERE season_year = @season_year) AND team_id = (SELECT team_id FROM team WHERE school_name = @team)

	--get field goal percentage
	DECLARE @field_goal_percentage DECIMAL(12,4)
	SELECT @field_goal_percentage = sum_field_goals_made/sum_field_goal_attempts
	FROM field_goal_totals
	WHERE player_first_name = @player_first_name AND player_last_name = @player_last_name AND season_id = @season_id

RETURN @field_goal_percentage
END
GO


