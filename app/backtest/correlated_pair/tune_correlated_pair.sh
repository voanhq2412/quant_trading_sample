#
# WEEKLY 
multipliers=(5 6 7 8 9 10 11 12 13 14 15)
deviations=(0.05 0.1 0.15 0.2)

# # ### MONTHLY
# multipliers=(2 2.5 3.5 4 4.5 5.5 6 6.5 7)
# deviations=(0.01 0.02 0.03 0.04)

portions=(0.1)
degrees=(1 2)

predictor="$1"
target="$2"

#### EXAMPLE 
# bash app/backtest/correlated_pair/tune_correlated_pair.sh DIG VCG

max_annualized_returns=-1

for m in "${multipliers[@]}"; do
    for p in "${portions[@]}"; do
        for d in "${deviations[@]}"; do
            for x in "${degrees[@]}"; do
                output=$(python3 -m app.backtest.correlated_pair.correlated_pair --pair "$predictor" "$target" --multiplier "$m" --max_portion "$p" --max_dev "$d" --degree "$x" --initial_capital 3000000 --backtest)
                echo "$output" 

                annualized_returns=$(echo "$output" | awk '/annualized_returns:/ {print $2}')
                if (( $(echo "$annualized_returns > $max_annualized_returns" | bc -l) )); then
                    max_annualized_returns=$annualized_returns
                fi                
                echo 
            done
        done
    done
done

echo "Maximum Annualized Returns for $predictor $target: $max_annualized_returns"
