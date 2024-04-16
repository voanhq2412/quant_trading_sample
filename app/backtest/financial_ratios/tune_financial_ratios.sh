multipliers=(0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2)
degrees=(1 2)
portions=(0.2 1.0)

max_annualized_returns=-1

for p in "${portions[@]}"; do
    for d in "${degrees[@]}"; do
        for m in "${multipliers[@]}"; do
            output=$(python3 -m app.backtest.financial_ratios.financial_ratios --ticker BAF --industry "Food & Beverage" --multiplier "$m" --degree "$d" --max_portion "$p" --initial_capital 3000000 --backtest)
            echo "$output" 

            annualized_returns=$(echo "$output" | awk '/annualized_returns:/ {print $2}')
            if (( $(echo "$annualized_returns > $max_annualized_returns" | bc -l) )); then
                max_annualized_returns=$annualized_returns
            fi                
            echo 
        done
    done
done

echo "Maximum Annualized Returns: $max_annualized_returns"
