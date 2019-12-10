package org.onnx4j.opsets.v6;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v5.OperatorSetSpecV5;
import org.onnx4j.opsets.v6.ops.DropoutV6;
import org.onnx4j.opsets.v6.ops.MulV6;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV6<T_TENSOR> extends OperatorSetSpecV5<T_TENSOR> {
	
	public abstract MulV6<T_TENSOR> getMulV6();
	
	public abstract DropoutV6<T_TENSOR> getDropoutV6();

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV5.super.initializeOperators();
		// 20191115
		operators.put(MulV6.OP_TYPE, this.getMulV6());
		// 20191120
		operators.put(DropoutV6.OP_TYPE, this.getDropoutV6());
		return operators;
	}

}
