package org.onnx4j.opsets.v3;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v2.OperatorSetSpecV2;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV3<T_TENSOR> extends OperatorSetSpecV2<T_TENSOR> {

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV2.super.initializeOperators();
		return operators;
	}

}
