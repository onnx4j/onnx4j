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
package org.onnx4j.opsets;

public class OperatorSets {
	
	private OperatorSet[] opsets;
	
	public static OperatorSets wrap(OperatorSet[] opsets) {
		return new OperatorSets(opsets);
	}
	
	private OperatorSets(OperatorSet[] opsets) {
		this.opsets = opsets;
	}
	
	/**
	 * 查找此版本实现下的Operator
	 * 
	 * @param opType
	 * @return
	 */
	public Operator getOperator(String opType) {
		for (OperatorSet opset : this.opsets) {
			Operator op = opset.getOp(opType);
			if (op == null)
				continue;
			
			return op;
		}
		
		return null;
	}

}