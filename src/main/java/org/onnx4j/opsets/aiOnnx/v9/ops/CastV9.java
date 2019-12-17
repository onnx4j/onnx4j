/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.aiOnnx.v9.ops;

import org.onnx4j.Inputs;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.aiOnnx.v6.ops.CastV6;
import org.onnx4j.opsets.aiOnnx.v9.AiOnnxOperatorV9;

/**
 * Cast
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Cast-9
 * @version This version of the operator has been available since version 9 of
 *          the default ONNX operator set.
 *
 */
public interface CastV9<T_TENSOR> extends CastV6<T_TENSOR>, AiOnnxOperatorV9 {

	class CastInputV9<T_TENSOR> extends CastInputV6<T_TENSOR> {

		public CastInputV9(Node node, Inputs inputs) {
			super(node, inputs);
		}

	}

	/**
	 * The operator casts the elements of a given input tensor to a data type
	 * specified by the 'to' argument and returns an output tensor of the same
	 * size in the converted type. The 'to' argument must be one of the data
	 * types specified in the 'DataType' enum field in the TensorProto message.
	 * 
	 * Casting from string tensor in plain (e.g., "3.14" and "1000") and
	 * scientific numeric representations (e.g., "1e-5" and "1E8") to float
	 * types is supported. For example, converting string "100.5" to an integer
	 * may result 100. There are some string literals reserved for special
	 * floating-point values; "+INF" (and "INF"), "-INF", and "NaN" are positive
	 * infinity, negative infinity, and not-a-number, respectively. Any string
	 * which can exactly match "+INF" in a case-insensitive way would be mapped
	 * to positive infinite. Similarly, this case-insensitive rule is applied to
	 * "INF" and "NaN". When casting from numeric tensors to string tensors,
	 * plain floating-point representation (such as "314.15926") would be used.
	 * Converting non-numerical-literal string such as "Hello World!" is an
	 * undefined behavior. Cases of converting string representing
	 * floating-point arithmetic value, such as "2.718", to INT is an undefined
	 * behavior.
	 * 
	 * Conversion from a numerical type to any numerical type is always allowed.
	 * User must be aware of precision loss and value change caused by range
	 * difference between two types. For example, a 64-bit float 3.1415926459
	 * may be round to a 32-bit float 3.141592. Similarly, converting an integer
	 * 36 to Boolean may produce 1 because we truncate bits which can't be
	 * stored in the targeted type.
	 * 
	 * @param t1
	 *            Input tensor to be cast.
	 * @param to
	 *            The data type to which the elements of the input tensor are
	 *            cast. Strictly must be one of the types from DataType enum in
	 *            TensorProto
	 * @return Output tensor with the same shape as input with type specified by
	 *         the 'to' argument
	 */
	public T_TENSOR cast(T_TENSOR t1, Long to);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		CastInputV9<T_TENSOR> input = new CastInputV9<T_TENSOR>(node, inputs);
		return Outputs.wrap(node, this.cast(input.getT1(), input.getToDTNumber()));
	}

}