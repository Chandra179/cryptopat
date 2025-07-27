#!/usr/bin/env python3

import asyncio
import logging
import sys
from cli.parser import create_parser
from cli.interactive import interactive_mode
from cli.commands import test_data_collection
from app.crypto_trend_app import CryptoTrendApp
from config.settings import VALID_METHODS, METHOD_REQUIREMENTS, TIMEFRAME_ORDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_method_requirements(methods, analysis_days, timeframe):
    """
    Validate that analysis configuration meets method requirements.
    
    Args:
        methods: List of analysis methods to validate
        analysis_days: Number of analysis days requested
        timeframe: Timeframe for analysis
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    for method in methods:
        if method not in METHOD_REQUIREMENTS:
            continue
            
        req = METHOD_REQUIREMENTS[method]
        
        # Check minimum analysis days
        if analysis_days < req['min_analysis_days']:
            errors.append(f"Method '{method}' requires minimum {req['min_analysis_days']} analysis days, got {analysis_days}")
        
        # Check minimum timeframe (equal or less granular timeframes allowed)
        method_tf_idx = TIMEFRAME_ORDER.index(req['min_timeframe'])
        try:
            current_tf_idx = TIMEFRAME_ORDER.index(timeframe)
            if current_tf_idx < method_tf_idx:
                errors.append(f"Method '{method}' requires minimum '{req['min_timeframe']}' granularity timeframe, got '{timeframe}'")
        except ValueError:
            errors.append(f"Invalid timeframe '{timeframe}' for method '{method}'")
    
    return errors

def main():
    """Main application entry point."""
    parser = create_parser()
    
    # If no arguments provided, start interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return
    
    args = parser.parse_args()
    
    # Show help if no command specified but args provided
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Handle different commands (for non-interactive mode)
    if args.command == 'collect':
        test_data_collection(args)
        return
    
    elif args.command == 'analyze':
        # Validate arguments for analyze command
        if args.predict_days <= 0:
            logger.error("Predict days must be positive")
            sys.exit(1)
        
        if args.analysis_days <= 0:
            logger.error("Analysis days must be positive")
            sys.exit(1)
        
        if args.analysis_days < args.predict_days:
            logger.warning(f"Analysis period ({args.analysis_days} days) is shorter than prediction period ({args.predict_days} days)")
        
        try:
            # Initialize application
            logger.info("Initializing CryptoPat application...")
            app = CryptoTrendApp(exchange=args.exchange)
            
            # Parse methods
            methods = [m.strip().lower() for m in args.methods.split(',') if m.strip()]
            methods = [m for m in methods if m in VALID_METHODS]
            
            if not methods:
                logger.error("No valid analysis methods specified. Use: sma, ema, ema_cross, macd, rsi or combinations")
                sys.exit(1)
            
            # Validate method requirements
            validation_errors = validate_method_requirements(methods, args.analysis_days, args.timeframe)
            if validation_errors:
                for error in validation_errors:
                    logger.error(error)
                sys.exit(1)
            
            # Run analysis
            asyncio.run(app.run_analysis(
                predict_days=args.predict_days,
                analysis_days=args.analysis_days,
                symbols=args.symbols,
                methods=methods,
                timeframe=args.timeframe
            ))
            
            logger.info("Analysis completed successfully")
            
        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Application error: {e}")
            sys.exit(1)
    
    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()